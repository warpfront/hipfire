// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Validated view of the Inkling-2 weights inside an HFQ file.
//!
//! Phase 1 stores physical weights once (24 layers) with per-expert tensors
//! for the routed MoE (128 experts × gate_up/down per layer). This module
//! indexes and validates that set by name/shape, then hands raw `quant_type`
//! + bytes to the forward/GPU upload layer. No numerics live here.

use hipfire_runtime::hfq::{HfqFile, HfqTensorInfo};
use std::collections::HashMap;

/// A source of decoded f32 tensors for the CPU forward. Implemented by the
/// quantized HFQ index and by the raw (unquantized) safetensors checkpoint.
pub trait WeightSource {
    fn f32(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>), String>;
}

/// A single raw HFQ tensor: metadata plus a copy of its encoded bytes.
#[derive(Clone)]
pub struct RawTensor {
    pub info: HfqTensorInfo,
    pub data: Vec<u8>,
}

/// Index of every Inkling-2 tensor in the trunk HFQ, keyed by verbatim HFQ name.
pub struct Inkling2Weights {
    pub tensors: HashMap<String, RawTensor>,
    /// Lazily decoded f32 weights (a 48-visit forward reuses each tensor many times).
    f32_cache: std::cell::RefCell<HashMap<String, (Vec<usize>, Vec<f32>)>>,
    /// `Some` iff the file carries the typed `qt44-padded-awq-v1` layout and
    /// [`certify_xts`] accepted it; `None` for the legacy (shipped RTN) layout.
    pub xts: Option<XtsCertificate>,
}

impl WeightSource for Inkling2Weights {
    fn f32(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>), String> {
        Inkling2Weights::f32(self, name)
    }
}

impl Inkling2Weights {
    /// Load and index all tensors, validating the expected physical-layer and
    /// per-expert structure against `cfg`.
    pub fn load(hfq: &HfqFile, cfg: &crate::config::Inkling2Config) -> Result<Self, String> {
        let mut tensors = HashMap::new();
        for info in hfq.tensor_infos() {
            let (i, data) = hfq
                .tensor_data_vec(&info.name)
                .ok_or_else(|| format!("inkling2: cannot read tensor {}", info.name))?;
            tensors.insert(
                info.name.clone(),
                RawTensor {
                    info: i.clone(),
                    data: data.to_vec(),
                },
            );
        }

        // Structural validation: every physical layer must carry its router,
        // norms and the full 128-expert set. Fail closed on a partial artifact.
        for il in 0..cfg.n_phys {
            let p = format!("model.llm.layers.{il}.");
            // Every layer has router, MLP norm and attention output norm;
            // SWA layers use `attn_norm`, full layers use three input norms.
            for suffix in ["mlp.gate.weight", "mlp_norm.weight"] {
                let name = format!("{p}{suffix}");
                if !tensors.contains_key(&name) {
                    return Err(format!("inkling2: missing {name}"));
                }
            }
            let attn_norm = if cfg.swa_layers[il] {
                "attn_norm.weight"
            } else {
                "q_input_norm.weight"
            };
            let name = format!("{p}{attn_norm}");
            if !tensors.contains_key(&name) {
                return Err(format!("inkling2: missing {name}"));
            }
            for e in 0..cfg.n_experts {
                for leaf in ["gate_up_proj.weight", "down_proj.weight"] {
                    let name = format!("{p}mlp.experts.{e}.{leaf}");
                    let t = tensors
                        .get(&name)
                        .ok_or_else(|| format!("inkling2: missing {name}"))?;
                    if t.info.shape.len() != 2 {
                        return Err(format!("inkling2: {name} not 2-D: {:?}", t.info.shape));
                    }
                }
            }
        }
        for name in ["model.llm.embed.weight", "model.llm.norm.weight"] {
            if !tensors.contains_key(name) {
                return Err(format!("inkling2: missing {name}"));
            }
        }

        // Typed layout: certify the XTS census, headers and sidecars on the
        // host before anything reaches the GPU. Legacy files skip this.
        let xts = if cfg.mq4_layout.is_xts() {
            Some(certify_xts(&tensors, cfg)?)
        } else {
            None
        };

        Ok(Self {
            tensors,
            f32_cache: std::cell::RefCell::new(HashMap::new()),
            xts,
        })
    }

    pub fn get(&self, name: &str) -> Option<&RawTensor> {
        self.tensors.get(name)
    }

    /// Decode a named weight to true (un-rotated) f32 values. Uses hipfire's
    /// public HFQ decoder, which inverse-FWHTs the MQ4G256/G256V2 formats so
    /// the result is the original weight matrix (row-major, `shape`).
    pub fn f32(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>), String> {
        if let Some(cached) = self.f32_cache.borrow().get(name) {
            return Ok(cached.clone());
        }
        let t = self
            .get(name)
            .ok_or_else(|| format!("inkling2: tensor not found: {name}"))?;
        let shape: Vec<usize> = t.info.shape.iter().map(|&d| d as usize).collect();
        let n: usize = shape.iter().product();
        let vals = hipfire_runtime::weight_backend::dequantize_weight_to_f32(
            t.info.quant_type,
            &t.data,
            n,
        );
        if vals.len() != n {
            return Err(format!(
                "inkling2: decoded {name} to {} values, expected {n}",
                vals.len()
            ));
        }
        let (shape, vals) = if let Some(cert) = self.xts.as_ref() {
            self.xts_calibrated_f32(name, cert, shape, vals)?
        } else {
            (shape, vals)
        };
        let result = (shape, vals);
        self.f32_cache
            .borrow_mut()
            .insert(name.to_string(), result.clone());
        Ok(result)
    }

    /// XTS layout: the stored qt44 of a calibrated tensor is `Q(R · (W ⊙ S))`,
    /// so the dequantized matrix is `W ⊙ S` per input column. Divide every
    /// column by its F16 sidecar value (as f32) to recover `W`; a down
    /// projection is additionally sliced from the physical `[m, 1280]` to the
    /// logical `[m, 1152]` the CPU forward consumes. Tensors without a sidecar
    /// (embed, unembed, router, keeps) are returned untouched.
    fn xts_calibrated_f32(
        &self,
        name: &str,
        cert: &XtsCertificate,
        shape: Vec<usize>,
        mut vals: Vec<f32>,
    ) -> Result<(Vec<usize>, Vec<f32>), String> {
        let Some(side) = self.get(&awq_sidecar_name(name)) else {
            return Ok((shape, vals));
        };
        if shape.len() != 2 {
            return Err(format!("inkling2 xts: {name} not 2-D: {shape:?}"));
        }
        let (m, k) = (shape[0], shape[1]);
        if side.data.len() != 2 * k {
            return Err(format!(
                "inkling2 xts: {name} sidecar has {} bytes, expected {}",
                side.data.len(),
                2 * k
            ));
        }
        let s: Vec<f32> = side
            .data
            .chunks_exact(2)
            .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect();
        for row in vals.chunks_exact_mut(k) {
            for (v, sc) in row.iter_mut().zip(&s) {
                *v /= *sc;
            }
        }
        if name.ends_with(".down_proj.weight") && k == cert.down_k_physical {
            let kl = cert.down_k_logical;
            let mut out = Vec::with_capacity(m * kl);
            for row in vals.chunks_exact(k) {
                out.extend_from_slice(&row[..kl]);
            }
            return Ok((vec![m, kl], out));
        }
        Ok((shape, vals))
    }

    /// Expected expert tensor count: n_phys × n_experts × 2 (gate_up + down).
    pub fn expected_expert_tensors(cfg: &crate::config::Inkling2Config) -> usize {
        cfg.n_phys * cfg.n_experts * 2
    }

    pub fn expert_tensor_count(&self) -> usize {
        self.tensors
            .keys()
            .filter(|n| {
                n.contains(".mlp.experts.")
                    && n.ends_with(".weight")
                    && !n.ends_with(AWQ_SIDECAR_SUFFIX)
            })
            .count()
    }
}

// ── XTS layout `qt44-padded-awq-v1`: certification ─────────────────────────
//
// The typed layout (`config::MQ4_LAYOUT_XTS`) is a hint only: every claim of
// `xts/CONTRACT.md` is re-derived here from the tensor bytes before upload.

/// `<stem>.awq_scale.weight` suffix of an AWQ sidecar tensor.
pub const AWQ_SIDECAR_SUFFIX: &str = ".awq_scale.weight";
const QT_F16: u8 = 1;
const QT_MQ4G256: u8 = 13;
const QT_MQ4G256V2: u8 = 44;
const MQ4V2_GROUP_BYTES: usize = 136;
const F16_ONE_BITS: u16 = 0x3C00;
/// Incumbent artifact A's tensor count; XTS adds one sidecar per calibrated tensor.
const MINI_A_TENSORS: usize = 6790;

/// Sidecar tensor name of a weight: `<stem>.awq_scale.weight`.
pub fn awq_sidecar_name(weight: &str) -> String {
    match weight.strip_suffix(".weight") {
        Some(stem) => format!("{stem}{AWQ_SIDECAR_SUFFIX}"),
        None => format!("{weight}{AWQ_SIDECAR_SUFFIX}"),
    }
}

/// Logical and physical K of the routed down projection: the SiTU width and
/// that width rounded up to a whole 256-group.
pub fn xts_down_dims(cfg: &crate::config::Inkling2Config) -> (usize, usize) {
    (cfg.inter, cfg.inter.next_multiple_of(256))
}

/// What the certifier established about one physical layer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct XtsLayerFacts {
    /// Every attention projection qt44 half is symmetric.
    pub dense_symmetric: bool,
    /// Every routed expert gate_up / down qt44 half is symmetric.
    pub experts_symmetric: bool,
    /// Attention sidecars of the layer (6).
    pub dense_sidecars: usize,
    /// Routed-expert sidecars of the layer (`2 * n_experts`).
    pub expert_sidecars: usize,
}

/// Result of [`certify_xts`]: host-side facts the GPU upload and the route
/// capability record are built from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct XtsCertificate {
    pub layers: Vec<XtsLayerFacts>,
    /// The standalone `unembed` qt44 is symmetric.
    pub unembed_symmetric: bool,
    /// Total sidecars (6,288 on Mini).
    pub awq_sidecars: usize,
    pub down_k_logical: usize,
    pub down_k_physical: usize,
}

impl XtsCertificate {
    /// Layers whose dense and routed qt44 halves are all symmetric.
    pub fn symmetric_layers(&self) -> usize {
        self.layers
            .iter()
            .filter(|l| l.dense_symmetric && l.experts_symmetric)
            .count()
    }
}

/// Check every 256-group of a qt44 stream is symmetric: per half `d >= +0`
/// (sign clear, finite), `z_bits == f16_bits(-8*d)` and `d == 0 => z == 0x8000`.
fn xts_check_symmetric(name: &str, data: &[u8]) -> Result<(), String> {
    if data.len() % MQ4V2_GROUP_BYTES != 0 {
        return Err(format!(
            "inkling2 xts: {name}: {} bytes is not a whole number of {MQ4V2_GROUP_BYTES}-byte qt44 groups",
            data.len()
        ));
    }
    for (g, grp) in data.chunks_exact(MQ4V2_GROUP_BYTES).enumerate() {
        for h in 0..2 {
            let d_bits = u16::from_le_bytes([grp[4 * h], grp[4 * h + 1]]);
            let z_bits = u16::from_le_bytes([grp[4 * h + 2], grp[4 * h + 3]]);
            let d = half::f16::from_bits(d_bits).to_f32();
            let ok = d_bits & 0x8000 == 0
                && d.is_finite()
                && z_bits == half::f16::from_f32(-8.0 * d).to_bits()
                && half::f16::from_bits(z_bits).to_f32().is_finite()
                && (d != 0.0 || z_bits == 0x8000);
            if !ok {
                return Err(format!(
                    "inkling2 xts: {name}: group {g} half {h} is not symmetric: d=0x{d_bits:04x} \
                     z=0x{z_bits:04x} expected z=0x{:04x} with d finite and >= +0",
                    half::f16::from_f32(-8.0 * d).to_bits()
                ));
            }
        }
    }
    Ok(())
}

/// Validate one calibrated tensor (qt44 `[m, k]`, symmetric, exact bytes) and its
/// F16 `[k]` sidecar (finite, > 0; for a down projection the tail
/// `down_logical..k` is exactly 1.0). Returns the sidecar bytes.
fn xts_check_calibrated<'a>(
    tensors: &'a HashMap<String, RawTensor>,
    name: &str,
    m: usize,
    k: usize,
    down_logical: Option<usize>,
) -> Result<&'a [u8], String> {
    let t = tensors
        .get(name)
        .ok_or_else(|| format!("inkling2 xts: missing {name}"))?;
    if t.info.quant_type != QT_MQ4G256V2 {
        return Err(format!(
            "inkling2 xts: {name}: quant_type {} but the layout requires symmetric qt44",
            t.info.quant_type
        ));
    }
    let shape: Vec<usize> = t.info.shape.iter().map(|&d| d as usize).collect();
    if shape != [m, k] {
        return Err(format!(
            "inkling2 xts: {name}: shape {shape:?}, expected [{m}, {k}]"
        ));
    }
    if k % 256 != 0 {
        return Err(format!(
            "inkling2 xts: {name}: K {k} is not a multiple of 256"
        ));
    }
    let want = m * (k / 256) * MQ4V2_GROUP_BYTES;
    if t.data.len() != want {
        return Err(format!(
            "inkling2 xts: {name}: {} bytes, expected {m} rows x {} B = {want} (row stride {} B)",
            t.data.len(),
            (k / 256) * MQ4V2_GROUP_BYTES,
            (k / 256) * MQ4V2_GROUP_BYTES
        ));
    }
    xts_check_symmetric(name, &t.data)?;
    let sn = awq_sidecar_name(name);
    let s = tensors
        .get(&sn)
        .ok_or_else(|| format!("inkling2 xts: {name}: AWQ sidecar {sn} is missing"))?;
    if s.info.quant_type != QT_F16 {
        return Err(format!(
            "inkling2 xts: {sn}: quant_type {} but the sidecar must be F16 (qt 1)",
            s.info.quant_type
        ));
    }
    let sshape: Vec<usize> = s.info.shape.iter().map(|&d| d as usize).collect();
    if sshape != [k] || s.data.len() != 2 * k {
        return Err(format!(
            "inkling2 xts: {sn}: shape {sshape:?} / {} bytes, expected [{k}] / {} bytes",
            s.data.len(),
            2 * k
        ));
    }
    for (i, c) in s.data.chunks_exact(2).enumerate() {
        let bits = u16::from_le_bytes([c[0], c[1]]);
        if bits & 0x7C00 == 0x7C00 {
            return Err(format!(
                "inkling2 xts: {sn}[{i}] = 0x{bits:04x} is not finite"
            ));
        }
        if bits & 0x8000 != 0 || bits & 0x7FFF == 0 {
            return Err(format!(
                "inkling2 xts: {sn}[{i}] = 0x{bits:04x} is not strictly positive"
            ));
        }
        if let Some(logical) = down_logical {
            if i >= logical && bits != F16_ONE_BITS {
                return Err(format!(
                    "inkling2 xts: {sn}[{i}] = 0x{bits:04x}: the down tail {logical}..{k} must be exactly F16 1.0 (0x{F16_ONE_BITS:04x})"
                ));
            }
        }
    }
    Ok(&s.data)
}

#[derive(Clone, Copy)]
struct XtsDims {
    e: usize,
    eattn: usize,
    ekv: usize,
    erel: usize,
    inter: usize,
    phys: usize,
    n_experts: usize,
    n_phys: usize,
    vocab: usize,
}

fn xts_certify_layer(
    tensors: &HashMap<String, RawTensor>,
    d: XtsDims,
    il: usize,
) -> Result<XtsLayerFacts, String> {
    use rayon::prelude::*;
    let p = format!("model.llm.layers.{il}.");
    let dense: [(&str, usize, usize); 6] = [
        ("attn.wq_du.weight", d.eattn, d.e),
        ("attn.wk_dv.weight", d.ekv, d.e),
        ("attn.wv_dv.weight", d.ekv, d.e),
        ("attn.wr_du.weight", d.erel, d.e),
        ("attn.gate_attn.weight", d.eattn, d.e),
        ("attn.wo_ud.weight", d.e, d.eattn),
    ];
    let mut q_side: Option<&[u8]> = None;
    for (suffix, m, k) in dense {
        let name = format!("{p}{suffix}");
        let side = xts_check_calibrated(tensors, &name, m, k, None)?;
        match suffix {
            "attn.wq_du.weight" => q_side = Some(side),
            "attn.wr_du.weight" | "attn.gate_attn.weight" => {
                if q_side != Some(side) {
                    return Err(format!(
                        "inkling2 xts: {name}: sidecar differs from {p}attn.wq_du (wq_du / wr_du / gate_attn share one Q producer and must be byte-identical)"
                    ));
                }
            }
            _ => {}
        }
    }

    let router = format!("{p}mlp.gate.weight");
    let rt = tensors
        .get(&router)
        .ok_or_else(|| format!("inkling2 xts: missing {router}"))?;
    let rshape: Vec<usize> = rt.info.shape.iter().map(|&x| x as usize).collect();
    if rt.info.quant_type != QT_F16
        || rshape != [d.n_experts, d.e]
        || rt.data.len() != 2 * d.n_experts * d.e
    {
        return Err(format!(
            "inkling2 xts: {router}: qt {} shape {rshape:?} {} bytes, expected F16 (qt 1) [{}, {}] ({} bytes)",
            rt.info.quant_type,
            rt.data.len(),
            d.n_experts,
            d.e,
            2 * d.n_experts * d.e
        ));
    }

    // One quant type per expert table: a mix is its own diagnostic.
    let mut qts: Vec<(usize, &str, u8)> = Vec::new();
    for ex in 0..d.n_experts {
        for leaf in ["gate_up_proj", "down_proj"] {
            let name = format!("{p}mlp.experts.{ex}.{leaf}.weight");
            let t = tensors
                .get(&name)
                .ok_or_else(|| format!("inkling2 xts: missing {name}"))?;
            qts.push((ex, leaf, t.info.quant_type));
        }
    }
    if let Some(&(ex, leaf, qt)) = qts.iter().find(|x| x.2 != QT_MQ4G256V2) {
        let mixed = qts.iter().any(|x| x.2 != qt);
        return Err(format!(
            "inkling2 xts: layer {il} routed-expert table {} (expert {ex} {leaf} is qt{qt}; the layout requires qt44 everywhere)",
            if mixed { "mixes quant types" } else { "is uniformly not qt44" }
        ));
    }
    let results: Vec<Result<(), String>> = (0..d.n_experts)
        .into_par_iter()
        .map(|ex| {
            xts_check_calibrated(
                tensors,
                &format!("{p}mlp.experts.{ex}.gate_up_proj.weight"),
                2 * d.inter,
                d.e,
                None,
            )?;
            xts_check_calibrated(
                tensors,
                &format!("{p}mlp.experts.{ex}.down_proj.weight"),
                d.e,
                d.phys,
                Some(d.inter),
            )?;
            Ok(())
        })
        .collect();
    results.into_iter().collect::<Result<Vec<()>, String>>()?;
    Ok(XtsLayerFacts {
        dense_symmetric: true,
        experts_symmetric: true,
        dense_sidecars: 6,
        expert_sidecars: 2 * d.n_experts,
    })
}

/// Certify the `qt44-padded-awq-v1` census, shapes/strides, symmetric headers
/// and sidecars of `tensors` against `cfg` (host only, rayon). A model-global
/// symmetric flag is never consulted: every qt44 half is read.
pub fn certify_xts(
    tensors: &HashMap<String, RawTensor>,
    cfg: &crate::config::Inkling2Config,
) -> Result<XtsCertificate, String> {
    use rayon::prelude::*;
    use std::collections::BTreeSet;
    let (down_logical, down_phys) = xts_down_dims(cfg);
    let d = XtsDims {
        e: cfg.n_embd,
        eattn: cfg.n_head * cfg.head_dim,
        ekv: cfg.n_head_kv * cfg.head_dim,
        erel: cfg.n_head * cfg.n_rel,
        inter: down_logical,
        phys: down_phys,
        n_experts: cfg.n_experts,
        n_phys: cfg.n_phys,
        vocab: cfg.vocab_size,
    };
    let layer_results: Vec<Result<XtsLayerFacts, String>> = (0..d.n_phys)
        .into_par_iter()
        .map(|il| xts_certify_layer(tensors, d, il))
        .collect();
    let mut layers = Vec::with_capacity(d.n_phys);
    for r in layer_results {
        layers.push(r?);
    }

    // Token IO: embed qt44 (incumbent affine bytes allowed, no sidecar),
    // unembed symmetric qt44.
    fn io<'a>(
        tensors: &'a HashMap<String, RawTensor>,
        d: XtsDims,
        name: &str,
    ) -> Result<&'a RawTensor, String> {
        let t = tensors
            .get(name)
            .ok_or_else(|| format!("inkling2 xts: missing {name}"))?;
        let shape: Vec<usize> = t.info.shape.iter().map(|&x| x as usize).collect();
        let want = d.vocab * (d.e / 256) * MQ4V2_GROUP_BYTES;
        if t.info.quant_type != QT_MQ4G256V2 || shape != [d.vocab, d.e] || t.data.len() != want {
            return Err(format!(
                "inkling2 xts: {name}: qt {} shape {shape:?} {} bytes, expected qt44 [{}, {}] ({want} bytes)",
                t.info.quant_type,
                t.data.len(),
                d.vocab,
                d.e
            ));
        }
        Ok(t)
    }
    io(tensors, d, "model.llm.embed.weight")?;
    let unembed = io(tensors, d, "model.llm.unembed.weight")?;
    xts_check_symmetric("model.llm.unembed.weight", &unembed.data)?;

    // Exact census: tensor classes and the sidecar name set.
    let n_calibrated = d.n_phys * (6 + 2 * d.n_experts);
    let mut qt44 = 0usize;
    let mut first_qt13: Option<&str> = None;
    let mut n_qt13 = 0usize;
    let mut actual_sidecars: BTreeSet<&str> = BTreeSet::new();
    for (name, t) in tensors {
        if name.ends_with(AWQ_SIDECAR_SUFFIX) {
            actual_sidecars.insert(name.as_str());
        } else if t.info.quant_type == QT_MQ4G256V2 {
            qt44 += 1;
        } else if t.info.quant_type == QT_MQ4G256 {
            n_qt13 += 1;
            first_qt13.get_or_insert(name.as_str());
        }
    }
    if n_qt13 != 0 {
        return Err(format!(
            "inkling2 xts: {n_qt13} qt13 tensors present (first {}); the layout has none",
            first_qt13.unwrap_or("?")
        ));
    }
    if qt44 != n_calibrated + 2 {
        return Err(format!(
            "inkling2 xts: {qt44} qt44 tensors, expected {} ({n_calibrated} calibrated + embed + unembed)",
            n_calibrated + 2
        ));
    }
    let mut expected_sidecars: BTreeSet<String> = BTreeSet::new();
    for il in 0..d.n_phys {
        let p = format!("model.llm.layers.{il}.");
        for s in [
            "attn.wq_du",
            "attn.wk_dv",
            "attn.wv_dv",
            "attn.wr_du",
            "attn.gate_attn",
            "attn.wo_ud",
        ] {
            expected_sidecars.insert(format!("{p}{s}{AWQ_SIDECAR_SUFFIX}"));
        }
        for ex in 0..d.n_experts {
            for leaf in ["gate_up_proj", "down_proj"] {
                expected_sidecars.insert(format!("{p}mlp.experts.{ex}.{leaf}{AWQ_SIDECAR_SUFFIX}"));
            }
        }
    }
    let extra: Vec<&str> = actual_sidecars
        .iter()
        .copied()
        .filter(|n| !expected_sidecars.contains(*n))
        .collect();
    if !extra.is_empty() {
        return Err(format!(
            "inkling2 xts: {} unexpected AWQ sidecar tensor(s), first {}",
            extra.len(),
            extra[0]
        ));
    }
    if actual_sidecars.len() != expected_sidecars.len() {
        let missing = expected_sidecars
            .iter()
            .find(|n| !actual_sidecars.contains(n.as_str()))
            .map(String::as_str)
            .unwrap_or("?");
        return Err(format!(
            "inkling2 xts: {} AWQ sidecars, expected {} (first missing {missing})",
            actual_sidecars.len(),
            expected_sidecars.len()
        ));
    }
    if d.n_phys == 24 && d.n_experts == 128 && tensors.len() != MINI_A_TENSORS + n_calibrated {
        return Err(format!(
            "inkling2 xts: {} tensors, expected {} (artifact A's {MINI_A_TENSORS} + {n_calibrated} sidecars)",
            tensors.len(),
            MINI_A_TENSORS + n_calibrated
        ));
    }
    Ok(XtsCertificate {
        layers,
        unembed_symmetric: true,
        awq_sidecars: n_calibrated,
        down_k_logical: down_logical,
        down_k_physical: down_phys,
    })
}

/// A weight source backed directly by the raw HF safetensors checkpoint
/// (BF16/F32, no quantization). Used for the decisive parity experiment:
/// run the CPU forward on unquantized weights and compare to llama.cpp.
pub struct RawWeights {
    tensors: HashMap<String, (Vec<usize>, Vec<f32>)>,
}

impl RawWeights {
    pub fn open(path: &std::path::Path) -> Result<Self, String> {
        let file = std::fs::File::open(path).map_err(|e| format!("open {path:?}: {e}"))?;
        let mmap = unsafe { memmap2::Mmap::map(&file).map_err(|e| format!("mmap: {e}"))? };
        let st = safetensors::SafeTensors::deserialize(&mmap).map_err(|e| format!("parse: {e}"))?;
        let mut tensors = HashMap::new();
        for (name, view) in st.tensors() {
            let shape: Vec<usize> = view.shape().to_vec();
            let n: usize = shape.iter().product();
            let vals: Vec<f32> = match view.dtype() {
                safetensors::Dtype::F32 => view
                    .data()
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
                safetensors::Dtype::BF16 => view
                    .data()
                    .chunks_exact(2)
                    .map(|c| {
                        let b = u16::from_le_bytes([c[0], c[1]]);
                        half::bf16::from_bits(b).to_f32()
                    })
                    .collect(),
                safetensors::Dtype::F16 => view
                    .data()
                    .chunks_exact(2)
                    .map(|c| {
                        let b = u16::from_le_bytes([c[0], c[1]]);
                        half::f16::from_bits(b).to_f32()
                    })
                    .collect(),
                other => return Err(format!("unsupported dtype {other} for {name}")),
            };
            if vals.len() != n {
                return Err(format!("{name}: {} values != {n}", vals.len()));
            }
            tensors.insert(name, (shape, vals));
        }
        Ok(Self { tensors })
    }
}

impl WeightSource for RawWeights {
    fn f32(&self, hfq_name: &str) -> Result<(Vec<usize>, Vec<f32>), String> {
        // Per-expert HFQ names `...experts.E.gate_up_proj.weight` /
        // `down_proj.weight` correspond to stacked checkpoint tensors
        // `...experts.w13_weight` [E, 2*inter, hidden] and
        // `...experts.w2_weight` [E, hidden, inter]. Slice out one expert and
        // regroup w13 interleaved [g0,u0,g1,u1,...] to [g...,u...].
        if let Some(rest) = hfq_name.strip_suffix(".gate_up_proj.weight") {
            if let Some(pos) = rest.rfind(".experts.") {
                let e: usize = rest[pos + ".experts.".len()..]
                    .parse()
                    .map_err(|_| "bad expert idx")?;
                let base = &rest[..pos + ".experts.".len()];
                let (shape, v) = self
                    .tensors
                    .get(&format!("{base}w13_weight"))
                    .ok_or("missing w13_weight")?;
                let ne = shape[0];
                let rows = shape[1];
                let cols = shape[2];
                let inter = rows / 2;
                let mut out = vec![0.0f32; rows * cols];
                let off = e * rows * cols;
                for j in 0..inter {
                    for c in 0..cols {
                        out[j * cols + c] = v[off + (2 * j) * cols + c];
                        out[(inter + j) * cols + c] = v[off + (2 * j + 1) * cols + c];
                    }
                }
                return Ok((vec![rows, cols], out));
            }
        }
        if let Some(rest) = hfq_name.strip_suffix(".down_proj.weight") {
            if let Some(pos) = rest.rfind(".experts.") {
                let e: usize = rest[pos + ".experts.".len()..]
                    .parse()
                    .map_err(|_| "bad expert idx")?;
                let base = &rest[..pos + ".experts.".len()];
                let (shape, v) = self
                    .tensors
                    .get(&format!("{base}w2_weight"))
                    .ok_or("missing w2_weight")?;
                let ne = shape[0];
                let per = shape[1] * shape[2];
                let out = v[e * per..(e + 1) * per].to_vec();
                let _ = ne;
                return Ok((vec![shape[1], shape[2]], out));
            }
        }
        self.tensors
            .get(hfq_name)
            .cloned()
            .ok_or_else(|| format!("raw: tensor not found: {hfq_name}"))
    }
}

/// Weight source backed by the F32 GGUF exported to a simple binary
/// (`gguf_to_f32_bin.py`): `[n][ (u32 name_len, name, u32 ndim, dims, f32 data) ]`.
/// All tensors are already dequantized F32 — the parity oracle's exact weights.
pub struct GgufF32 {
    path: std::path::PathBuf,
    /// name -> (shape, byte offset of data in file)
    index: HashMap<String, (Vec<usize>, u64)>,
    cache: std::cell::RefCell<HashMap<String, (Vec<usize>, Vec<f32>)>>,
}

impl GgufF32 {
    pub fn open(path: &std::path::Path) -> Result<Self, String> {
        use std::io::{Read, Seek, SeekFrom};
        let mut f = std::fs::File::open(path).map_err(|e| format!("open: {e}"))?;
        let mut buf = [0u8; 4];
        f.read_exact(&mut buf)
            .map_err(|e| format!("read count: {e}"))?;
        let n = u32::from_le_bytes(buf);
        let mut index = HashMap::new();
        for _ in 0..n {
            f.read_exact(&mut buf)
                .map_err(|e| format!("read name len: {e}"))?;
            let nl = u32::from_le_bytes(buf) as usize;
            let mut nb = vec![0u8; nl];
            f.read_exact(&mut nb)
                .map_err(|e| format!("read name: {e}"))?;
            let name = String::from_utf8(nb).map_err(|e| format!("name: {e}"))?;
            f.read_exact(&mut buf)
                .map_err(|e| format!("read ndim: {e}"))?;
            let nd = u32::from_le_bytes(buf) as usize;
            let mut db = vec![0u8; nd * 4];
            f.read_exact(&mut db)
                .map_err(|e| format!("read dims: {e}"))?;
            let shape: Vec<usize> = db
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize)
                .collect();
            let len: usize = shape.iter().product();
            let off = f.stream_position().map_err(|e| format!("pos: {e}"))?;
            f.seek(SeekFrom::Current((len * 4) as i64))
                .map_err(|e| format!("seek: {e}"))?;
            index.insert(name, (shape, off));
        }
        Ok(Self {
            path: path.to_path_buf(),
            index,
            cache: std::cell::RefCell::new(HashMap::new()),
        })
    }

    fn read(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>), String> {
        use std::io::{Read, Seek, SeekFrom};
        let (shape, off) = self
            .index
            .get(name)
            .cloned()
            .ok_or_else(|| format!("gguf: missing {name}"))?;
        let len: usize = shape.iter().product();
        let mut f = std::fs::File::open(&self.path).map_err(|e| format!("open: {e}"))?;
        f.seek(SeekFrom::Start(off))
            .map_err(|e| format!("seek: {e}"))?;
        // The .bin carries file-order dim labels but torch-order bytes; the
        // torch view is flat.reshape(reversed(shape)). Store the reversed
        // (torch-logical) shape so all downstream name handling is uniform.
        let shape: Vec<usize> = shape.iter().rev().copied().collect();
        // Read directly into f32 storage (avoids a second multi-GB byte buffer).
        let mut vals = vec![0.0f32; len];
        {
            let bytes =
                unsafe { std::slice::from_raw_parts_mut(vals.as_mut_ptr() as *mut u8, len * 4) };
            f.read_exact(bytes)
                .map_err(|e| format!("read {name}: {e}"))?;
        }
        Ok((shape, vals))
    }
}

/// The cross-pass `q` name is the one that is not the `kv` name; both end in
/// `_queries`, so a suffix test on `_queries` alone classifies both as q.
fn is_cross_pass_q_query(name: &str) -> bool {
    !name.ends_with("kv_queries")
}

impl WeightSource for GgufF32 {
    fn f32(&self, hfq: &str) -> Result<(Vec<usize>, Vec<f32>), String> {
        // map HFQ (checkpoint) name -> GGUF name, with per-GGUF-name cache.
        let get = |n: &str| -> Result<(Vec<usize>, Vec<f32>), String> {
            if let Some(v) = self.cache.borrow().get(n) {
                return Ok(v.clone());
            }
            let v = self.read(n)?;
            // Only cache small tensors (norms, convs, clips, controllers,
            // small projections). Embed/head (~1.8 GB) and expert weights
            // (~21 MB × hundreds) would exhaust RAM if cached.
            if v.1.len() <= 4 * 1024 * 1024 {
                self.cache.borrow_mut().insert(n.to_string(), v.clone());
            }
            Ok(v)
        };
        if hfq == "model.llm.embed.weight" {
            return get("token_embd.weight");
        }
        if hfq == "model.llm.embed_norm.weight" {
            return self
                .read("token_embd_norm.weight")
                .or_else(|_| self.read("token_norm.weight"));
        }
        if hfq == "model.llm.norm.weight" {
            return get("output_norm.weight");
        }
        if hfq == "model.llm.unembed.weight" {
            return get("output.weight");
        }

        // per-expert
        if let Some(rest) = hfq.strip_prefix("model.llm.layers.") {
            // layers.N....
            let dot = rest.find('.').ok_or("bad layer name")?;
            let layer: usize = rest[..dot].parse().map_err(|_| "bad layer")?;
            let suffix = &rest[dot + 1..];
            let b = format!("blk.{layer}.");
            if let Some(e) = suffix.strip_prefix("mlp.experts.") {
                // mlp.experts.E.gate_up_proj.weight / down_proj.weight
                let dot2 = e.find('.').ok_or("bad expert")?;
                let ex: usize = e[..dot2].parse().map_err(|_| "bad expert idx")?;
                if e.ends_with("gate_up_proj.weight") {
                    // dims reversed at read -> torch [E, 2*inter, hidden].
                    let (shape, v) = get(&format!("{b}ffn_gate_up_exps.weight"))?;
                    let per = shape[1] * shape[2];
                    let out = v[ex * per..(ex + 1) * per].to_vec();
                    return Ok((vec![shape[1], shape[2]], out));
                }
                if e.ends_with("down_proj.weight") {
                    // torch [E, hidden, inter].
                    let (shape, v) = get(&format!("{b}ffn_down_exps.weight"))?;
                    let per = shape[1] * shape[2];
                    let out = v[ex * per..(ex + 1) * per].to_vec();
                    return Ok((vec![shape[1], shape[2]], out));
                }
            }
            // rel bias table: GGUF torch view [n_extent, n_rel]; forward wants
            // [n_rel, n_extent] (HFQ order) indexed [r * n_extent + d].
            if suffix == "attn.rel_logits_proj.proj" {
                let (shape, v) = get(&format!("{b}attn_rel_bias.weight"))?;
                let (extent, nrel) = (shape[0], shape[1]);
                let mut t = vec![0.0f32; nrel * extent];
                for r in 0..nrel {
                    for d2 in 0..extent {
                        t[r * extent + d2] = v[d2 * nrel + r];
                    }
                }
                return Ok((vec![nrel, extent], t));
            }
            // controllers.{attn,k,v,mlp}.{down,gain,forget}_weight
            // GGUF stores them transposed vs hipfire's control() convention.
            for (short, tag) in [("attn", "q"), ("k", "k"), ("v", "v"), ("mlp", "ffn")] {
                for (kind, ktag) in [("down", "down"), ("gain", "gain"), ("forget", "forget")] {
                    if suffix == format!("controllers.{short}.{kind}_weight") {
                        // dims reversed at read -> torch order, no transpose.
                        return get(&format!("{b}loop_{tag}_ctrl_{ktag}.weight"));
                    }
                }
            }
            let map = |s: &str| -> String {
                match s {
                    "attn.wq_du.weight" => format!("{b}attn_q.weight"),
                    "attn.wk_dv.weight" => format!("{b}attn_k.weight"),
                    "attn.wv_dv.weight" => format!("{b}attn_v.weight"),
                    "attn.wr_du.weight" => format!("{b}attn_rel.weight"),
                    "attn.gate_attn.weight" => format!("{b}attn_gate.weight"),
                    "attn.wo_ud.weight" => format!("{b}attn_output.weight"),
                    "attn.attn_sink_proj.weight" => format!("{b}attn_sink_proj.weight"),
                    "attn.rel_logits_proj.proj" => format!("{b}attn_rel_bias.weight"),
                    "attn.q_l2_clip.scale" => format!("{b}attn_q_clip.weight"),
                    "attn.rel_l2_clip.scale" => format!("{b}attn_rel_clip.weight"),
                    "attn_norm.weight" => format!("{b}attn_norm.weight"),
                    "q_input_norm.weight" => format!("{b}attn_q_inp_norm.weight"),
                    "k_input_norm.weight" => format!("{b}attn_k_inp_norm.weight"),
                    "v_input_norm.weight" => format!("{b}attn_v_inp_norm.weight"),
                    "attn.k_sconv.weight" | "attn.k_sconv.static_w" => {
                        format!("{b}attn_k_conv.weight")
                    }
                    "attn.v_sconv.weight" | "attn.v_sconv.static_w" => {
                        format!("{b}attn_v_conv.weight")
                    }
                    "attn.k_sconv.weight_proj" => format!("{b}attn_k_conv_proj.weight"),
                    "attn.v_sconv.weight_proj" => format!("{b}attn_v_conv_proj.weight"),
                    "attn_sconv.weight" => format!("{b}attn_post_conv.weight"),
                    "mlp_sconv.weight" => format!("{b}ffn_post_conv.weight"),
                    "mlp.gate.weight" => format!("{b}ffn_gate_inp.weight"),
                    "mlp.gate.bias" => format!("{b}exp_probs_b.bias"),
                    "mlp_norm.weight" => format!("{b}ffn_norm.weight"),
                    _ => format!("{b}{s}"),
                }
            };
            return get(&map(suffix));
        }
        // Cross-pass globals: GGUF stores per-layer loop_msg_query [n_embd,4]
        // for core layers 6..18. Reassemble the two (24,n_embd) globals that
        // pack_message_queries expects: q rows = [msg[..0], msg[..3]], kv rows
        // = [msg[..1], msg[..2]] per core layer.
        if hfq == "model.llm.middle_cycle_cross_pass_queries"
            || hfq == "model.llm.middle_cycle_cross_pass_kv_queries"
        {
            // Both names end in "_queries"; only the kv one is not q.
            let is_q = is_cross_pass_q_query(hfq);
            let e = 2304usize;
            let mut q_rows: Vec<f32> = Vec::with_capacity(24 * e);
            let mut kv_rows: Vec<f32> = Vec::with_capacity(24 * e);
            for layer in 6..18 {
                let (_, m) = get(&format!("blk.{layer}.loop_msg_query.weight"))?;
                // m layout [n_embd, 4] row-major => col c at m[c*e..(c+1)*e]
                q_rows.extend_from_slice(&m[0..e]);
                q_rows.extend_from_slice(&m[3 * e..4 * e]);
                kv_rows.extend_from_slice(&m[1 * e..2 * e]);
                kv_rows.extend_from_slice(&m[2 * e..3 * e]);
            }
            return Ok((vec![24, e], if is_q { q_rows } else { kv_rows }));
        }
        // AttnRes queries: dims reversed at read -> torch [rows, n_embd] (or
        // [n_embd] for single-row). GGUF query i lives on block i+1.
        if hfq == "model.llm.attn_res.output_query" {
            return get("output_res_query.weight");
        }
        if let Some(rest) = hfq.strip_prefix("model.llm.attn_res.attn_queries.") {
            let i: usize = rest.parse().map_err(|_| "bad query idx")?;
            return get(&format!("blk.{}.attn_res_query.weight", i + 1));
        }
        if let Some(rest) = hfq.strip_prefix("model.llm.attn_res.mlp_queries.") {
            let i: usize = rest.parse().map_err(|_| "bad query idx")?;
            return get(&format!("blk.{i}.ffn_res_query.weight"));
        }
        get(hfq)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Writes a synthetic gguf-f32.bin carrying the AttnRes/cross-pass tensors
    /// so the HFQ-name -> GGUF-name mapping is testable without the 98 GB
    /// export. This is the regression net for the class of bug where a name
    /// went unmapped and the forward silently skipped a required tensor.
    fn synthetic_bin(entries: &[(&str, &[usize])]) -> std::path::PathBuf {
        use std::io::Write;
        let path = std::env::temp_dir().join(format!(
            "inkling2-synthetic-{}-{}.bin",
            std::process::id(),
            entries.len()
        ));
        let mut f = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
        f.write_all(&(entries.len() as u32).to_le_bytes()).unwrap();
        for (name, dims) in entries {
            let len: usize = dims.iter().product();
            f.write_all(&(name.len() as u32).to_le_bytes()).unwrap();
            f.write_all(name.as_bytes()).unwrap();
            f.write_all(&(dims.len() as u32).to_le_bytes()).unwrap();
            for d in *dims {
                f.write_all(&(*d as u32).to_le_bytes()).unwrap();
            }
            for i in 0..len {
                f.write_all(&((i as f32) * 0.5).to_le_bytes()).unwrap();
            }
        }
        f.flush().unwrap();
        path
    }

    #[test]
    fn gguf_mapping_covers_attnres_and_cross_pass_tensors() {
        let e = 2304usize;
        // GGUF file order is ne0 first, so a [4, e] torch view is stored [e, 4].
        let one = [e];
        let four = [e, 4];
        let mut names: Vec<(String, usize)> = vec![
            ("output_res_query.weight".to_string(), 0),
            ("output_norm.weight".to_string(), 0),
            ("blk.6.attn_res_query.weight".to_string(), 0),
            ("blk.6.ffn_res_query.weight".to_string(), 0),
        ];
        for l in 6..18 {
            names.push((format!("blk.{l}.loop_msg_query.weight"), 1));
        }
        let entries: Vec<(&str, &[usize])> = names
            .iter()
            .map(|(n, four_col)| {
                (
                    n.as_str(),
                    if *four_col == 1 { &four[..] } else { &one[..] },
                )
            })
            .collect();
        let path = synthetic_bin(&entries);
        let w = GgufF32::open(&path).expect("open synthetic");
        let len = |name: &str| -> usize {
            w.f32(name)
                .unwrap_or_else(|e| panic!("{name} must map: {e}"))
                .1
                .len()
        };
        // bug #4: this one was unmapped, so the final AttnRes mix was skipped
        assert_eq!(len("model.llm.attn_res.output_query"), e);
        assert_eq!(len("model.llm.attn_res.attn_queries.5"), e);
        assert_eq!(len("model.llm.attn_res.mlp_queries.6"), e);
        let q = w
            .f32("model.llm.middle_cycle_cross_pass_queries")
            .unwrap()
            .1;
        let kv = w
            .f32("model.llm.middle_cycle_cross_pass_kv_queries")
            .unwrap()
            .1;
        assert_eq!(q.len(), 24 * e);
        assert_eq!(kv.len(), 24 * e);
        // the two globals must not be the same rows (bug #3)
        assert_ne!(q, kv);
        // unknown names still fail closed
        assert!(w.f32("model.llm.attn_res.nonexistent").is_err());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn cross_pass_query_names_classify_q_and_kv() {
        assert!(is_cross_pass_q_query(
            "model.llm.middle_cycle_cross_pass_queries"
        ));
        assert!(is_cross_pass_q_query("model.cross_pass_messages.queries"));
        assert!(!is_cross_pass_q_query(
            "model.llm.middle_cycle_cross_pass_kv_queries"
        ));
        assert!(!is_cross_pass_q_query(
            "model.cross_pass_messages.kv_queries"
        ));
    }

    #[test]
    fn expected_expert_count_for_mini() {
        let cfg = crate::config::LoopConfig::mini();
        // Use the typed config shape indirectly: 24 phys × 128 experts × 2.
        assert_eq!(
            24 * 128 * 2,
            Inkling2Weights::expected_expert_tensors(&tiny_cfg())
        );
        let _ = cfg;
    }

    fn tiny_cfg() -> crate::config::Inkling2Config {
        // Reuse the metadata parser test's config by parsing the same JSON.
        crate::config::Inkling2Config::from_metadata_json(TEST_META).unwrap()
    }

    pub(crate) fn test_meta() -> &'static str {
        TEST_META
    }

    const TEST_META: &str = r#"{"config":{"eos_token_id":200006,"text_config":{
        "num_hidden_layers":24,"hidden_size":2304,"vocab_size":201024,
        "unpadded_vocab_size":200058,"num_attention_heads":24,"num_key_value_heads":8,
        "head_dim":128,"sliding_window_size":1024,"d_rel":16,"rel_extent":1024,
        "qk_l2_clip_threshold":22.627416997969522,"rel_l2_clip_threshold":8.0,
        "log_scaling_n_floor":8192,"log_scaling_alpha":0.1,"sconv_kernel_size":4,
        "n_layers_per_attn_res_block":6,"final_logit_softcapping":30.0,
        "n_routed_experts":128,"num_experts_per_tok":8,"intermediate_size":1152,
        "route_scale":8.0,"mlp_gate_softcap":4.0,"mlp_up_softcap":25.0,"rms_norm_eps":1e-6,
        "local_layer_ids":[0,1,2,3,4,6,7,8,9,10,12,13,14,15,16,18,19,20,21,22],
        "loop_config":{"core_start":6,"core_end":18,"num_passes":3,"gate_rank":64,"is_varloop":false}
    }}}"#;

    #[test]
    #[ignore = "requires the 14 GB artifact; verify embedded tokenizer"]
    fn tokenizer_from_artifact() {
        use hipfire_runtime::hfq::HfqFile;
        let hfq = HfqFile::open(std::path::Path::new(
            r"D:\hipfire-models\inkling2-mini-mq4.hfq",
        ))
        .expect("open");
        let tok = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
            .expect("tokenizer from HFQ metadata");
        let ids = tok.encode("The capital of France is");
        eprintln!(
            "tokenized {} -> {} ids: {:?}",
            "The capital of France is",
            ids.len(),
            &ids[..ids.len().min(16)]
        );
        assert!(!ids.is_empty());
        let roundtrip = tok.decode(&ids);
        assert!(!roundtrip.is_empty());
        eprintln!("roundtrip: {:?}", &roundtrip[..roundtrip.len().min(80)]);
    }

    #[test]
    #[ignore = "print decoded qt44 stats to compare with GGUF"]
    fn inspect_qt44_decode() {
        use hipfire_runtime::hfq::HfqFile;
        let hfq = HfqFile::open(std::path::Path::new(
            r"D:\hipfire-models\inkling2-mini-mq4.hfq",
        ))
        .expect("open");
        let w = Inkling2Weights::load(
            &hfq,
            &crate::config::Inkling2Config::from_metadata_json(crate::TEST_METADATA).unwrap(),
        )
        .unwrap();
        for name in [
            "model.llm.layers.0.attn.wk_dv.weight",
            "model.llm.layers.0.attn.wq_du.weight",
        ] {
            let (shape, v) = w.f32(name).unwrap();
            let min = v.iter().copied().fold(f32::INFINITY, f32::min);
            let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mean: f64 = v.iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64;
            eprintln!("{name}: shape={shape:?} min={min} max={max} mean={mean}");
            let first: Vec<String> = v[..8].iter().map(|x| format!("{:.5}", x)).collect();
            eprintln!("  first8={:?}", first);
            if name.ends_with("wk_dv.weight") {
                let mut s = String::new();
                for x in &v {
                    s.push_str(&format!("{}\n", x.to_bits()));
                }
                std::fs::write(r"D:\hipfire-models\decoded-wk.txt", s).unwrap();
            }
        }
    }

    #[test]
    #[ignore = "requires the 14 GB inkling2-mini-mq4.hfq artifact"]
    fn loads_real_artifact() {
        let hfq = HfqFile::open(std::path::Path::new(
            r"D:\hipfire-models\inkling2-mini-mq4.hfq",
        ))
        .expect("open artifact");
        let cfg = tiny_cfg();
        let w = Inkling2Weights::load(&hfq, &cfg).expect("load weights");
        assert_eq!(
            w.expert_tensor_count(),
            Inkling2Weights::expected_expert_tensors(&cfg)
        );
        assert!(w.get("model.llm.embed.weight").is_some());
    }
}

#[cfg(test)]
pub(crate) mod xts_tests {
    use super::*;
    use crate::config::{Inkling2Config, Mq4Layout};

    /// Tiny geometry: 2 layers, 2 experts, hidden 256, inter 128 (down K 128 -> 256).
    pub(crate) fn tiny_cfg() -> Inkling2Config {
        let mut cfg = Inkling2Config::from_metadata_json(crate::TEST_METADATA).unwrap();
        cfg.n_phys = 2;
        cfg.n_embd = 256;
        cfg.n_head = 2;
        cfg.n_head_kv = 1;
        cfg.head_dim = 128;
        cfg.n_rel = 2;
        cfg.n_experts = 2;
        cfg.inter = 128;
        cfg.vocab_size = 4;
        cfg.swa_layers = vec![true, false];
        cfg.mq4_layout = Mq4Layout::Qt44PaddedAwqV1;
        cfg
    }

    fn f16b(v: f32) -> u16 {
        half::f16::from_f32(v).to_bits()
    }

    /// `m` rows of `k` weights as symmetric qt44 groups (every 5th half has d == 0).
    pub(crate) fn sym_qt44(m: usize, k: usize, salt: usize) -> Vec<u8> {
        let mut out = Vec::new();
        for g in 0..m * (k / 256) {
            let mut grp = [0u8; 136];
            for h in 0..2 {
                let d = if (g + h + salt) % 5 == 0 {
                    0.0
                } else {
                    ((g + salt) % 7 + 1) as f32 * 0.01
                };
                let d_bits = f16b(d);
                let dv = half::f16::from_bits(d_bits).to_f32();
                let z_bits = f16b(-8.0 * dv);
                grp[4 * h..4 * h + 2].copy_from_slice(&d_bits.to_le_bytes());
                grp[4 * h + 2..4 * h + 4].copy_from_slice(&z_bits.to_le_bytes());
            }
            for i in 0..128 {
                grp[8 + i] = ((i * 7 + g + salt) % 16) as u8 | ((((i * 3 + g) % 16) as u8) << 4);
            }
            out.extend_from_slice(&grp);
        }
        out
    }

    /// Affine (incumbent-style) qt44 groups: z = 1.0 regardless of d.
    fn affine_qt44(m: usize, k: usize) -> Vec<u8> {
        let mut out = sym_qt44(m, k, 1);
        for grp in out.chunks_exact_mut(136) {
            for h in 0..2 {
                grp[4 * h + 2..4 * h + 4].copy_from_slice(&f16b(1.0).to_le_bytes());
            }
        }
        out
    }

    pub(crate) fn f16_vec(vals: &[f32]) -> Vec<u8> {
        vals.iter().flat_map(|&v| f16b(v).to_le_bytes()).collect()
    }

    pub(crate) fn put(
        map: &mut HashMap<String, RawTensor>,
        name: &str,
        qt: u8,
        shape: &[usize],
        data: Vec<u8>,
    ) {
        map.insert(
            name.to_string(),
            RawTensor {
                info: HfqTensorInfo {
                    name: name.to_string(),
                    quant_type: qt,
                    shape: shape.iter().map(|&d| d as u32).collect(),
                    group_size: 256,
                    data_offset: 0,
                    data_size: data.len(),
                },
                data,
            },
        );
    }

    /// Sidecar `[k]`: 2.0 on the live prefix, exact 1.0 on the down tail.
    fn sidecar(k: usize, live: usize) -> Vec<u8> {
        let v: Vec<f32> = (0..k).map(|i| if i < live { 2.0 } else { 1.0 }).collect();
        f16_vec(&v)
    }

    fn put_calibrated(
        map: &mut HashMap<String, RawTensor>,
        name: &str,
        m: usize,
        k: usize,
        live: usize,
    ) {
        put(map, name, 44, &[m, k], sym_qt44(m, k, name.len()));
        put(map, &awq_sidecar_name(name), 1, &[k], sidecar(k, live));
    }

    pub(crate) fn build(cfg: &Inkling2Config) -> HashMap<String, RawTensor> {
        let (e, eattn, ekv, erel) = (
            cfg.n_embd,
            cfg.n_head * cfg.head_dim,
            cfg.n_head_kv * cfg.head_dim,
            cfg.n_head * cfg.n_rel,
        );
        let (logical, phys) = xts_down_dims(cfg);
        let mut map = HashMap::new();
        for il in 0..cfg.n_phys {
            let p = format!("model.llm.layers.{il}.");
            for (s, m, k) in [
                ("attn.wq_du", eattn, e),
                ("attn.wk_dv", ekv, e),
                ("attn.wv_dv", ekv, e),
                ("attn.wr_du", erel, e),
                ("attn.gate_attn", eattn, e),
                ("attn.wo_ud", e, eattn),
            ] {
                put_calibrated(&mut map, &format!("{p}{s}.weight"), m, k, k);
            }
            // wq_du / wr_du / gate_attn share one sidecar: copy wq's bytes.
            let q = map[&format!("{p}attn.wq_du.awq_scale.weight")].data.clone();
            for s in ["attn.wr_du", "attn.gate_attn"] {
                map.get_mut(&format!("{p}{s}.awq_scale.weight"))
                    .unwrap()
                    .data = q.clone();
            }
            put(
                &mut map,
                &format!("{p}mlp.gate.weight"),
                1,
                &[cfg.n_experts, e],
                f16_vec(&vec![0.5; cfg.n_experts * e]),
            );
            for ex in 0..cfg.n_experts {
                put_calibrated(
                    &mut map,
                    &format!("{p}mlp.experts.{ex}.gate_up_proj.weight"),
                    2 * cfg.inter,
                    e,
                    e,
                );
                put_calibrated(
                    &mut map,
                    &format!("{p}mlp.experts.{ex}.down_proj.weight"),
                    e,
                    phys,
                    logical,
                );
            }
        }
        put(
            &mut map,
            "model.llm.embed.weight",
            44,
            &[cfg.vocab_size, e],
            affine_qt44(cfg.vocab_size, e),
        );
        put(
            &mut map,
            "model.llm.unembed.weight",
            44,
            &[cfg.vocab_size, e],
            sym_qt44(cfg.vocab_size, e, 3),
        );
        map
    }

    fn err_of(map: &HashMap<String, RawTensor>, cfg: &Inkling2Config) -> String {
        certify_xts(map, cfg)
            .err()
            .expect("certification must fail")
    }

    const GU: &str = "model.llm.layers.1.mlp.experts.1.gate_up_proj.weight";
    const DN: &str = "model.llm.layers.0.mlp.experts.0.down_proj.weight";

    #[test]
    fn valid_tiny_layout_certifies() {
        let cfg = tiny_cfg();
        let cert = certify_xts(&build(&cfg), &cfg).unwrap();
        assert_eq!(cert.symmetric_layers(), 2);
        assert_eq!(cert.awq_sidecars, 2 * (6 + 2 * 2));
        assert_eq!((cert.down_k_logical, cert.down_k_physical), (128, 256));
        assert!(cert.unembed_symmetric);
        assert!(cert
            .layers
            .iter()
            .all(|l| l.dense_sidecars == 6 && l.expert_sidecars == 4));
    }

    #[test]
    fn planted_asymmetric_half_is_rejected_with_tensor_group_half() {
        let cfg = tiny_cfg();
        let mut map = build(&cfg);
        // Group 1 half 1 of an expert: z no longer -8*d.
        let t = map.get_mut(GU).unwrap();
        let off = 136 + 4 + 2;
        t.data[off..off + 2].copy_from_slice(&f16b(0.125).to_le_bytes());
        t.data[136 + 4..136 + 6].copy_from_slice(&f16b(0.25).to_le_bytes());
        let e = err_of(&map, &cfg);
        assert!(
            e.contains(GU) && e.contains("group 1 half 1") && e.contains("not symmetric"),
            "{e}"
        );
    }

    #[test]
    fn negative_zero_and_nonfinite_scales_and_unembed_are_rejected() {
        let cfg = tiny_cfg();
        // d == -0.0
        let mut map = build(&cfg);
        map.get_mut(GU).unwrap().data[0..2].copy_from_slice(&0x8000u16.to_le_bytes());
        assert!(err_of(&map, &cfg).contains("not symmetric"));
        // d == 0 but z != 0x8000
        let mut map = build(&cfg);
        let t = map.get_mut(GU).unwrap();
        t.data[0..2].copy_from_slice(&0u16.to_le_bytes());
        t.data[2..4].copy_from_slice(&0u16.to_le_bytes());
        assert!(err_of(&map, &cfg).contains("not symmetric"));
        // d == +inf
        let mut map = build(&cfg);
        let t = map.get_mut(GU).unwrap();
        t.data[0..2].copy_from_slice(&0x7C00u16.to_le_bytes());
        assert!(err_of(&map, &cfg).contains("not symmetric"));
        // asymmetric unembed
        let mut map = build(&cfg);
        map.insert("model.llm.unembed.weight".into(), {
            let mut t = map["model.llm.unembed.weight"].clone();
            t.data = affine_qt44(cfg.vocab_size, cfg.n_embd);
            t
        });
        let e = err_of(&map, &cfg);
        assert!(
            e.contains("model.llm.unembed.weight") && e.contains("not symmetric"),
            "{e}"
        );
    }

    #[test]
    fn missing_and_extra_sidecars_are_rejected() {
        let cfg = tiny_cfg();
        let mut map = build(&cfg);
        map.remove(&awq_sidecar_name(GU));
        let e = err_of(&map, &cfg);
        assert!(
            e.contains("is missing") && e.contains(&awq_sidecar_name(GU)),
            "{e}"
        );

        let mut map = build(&cfg);
        // A router (or any non-calibrated tensor) must not carry a sidecar.
        put(
            &mut map,
            "model.llm.layers.0.mlp.gate.awq_scale.weight",
            1,
            &[256],
            sidecar(256, 256),
        );
        let e = err_of(&map, &cfg);
        assert!(
            e.contains("unexpected AWQ sidecar") && e.contains("mlp.gate.awq_scale"),
            "{e}"
        );
    }

    #[test]
    fn bad_sidecar_values_shape_and_type_are_rejected() {
        let cfg = tiny_cfg();
        let sn = awq_sidecar_name(GU);
        for (bits, needle) in [
            (0x0000u16, "strictly positive"),
            (0xBC00, "strictly positive"),
            (0x7E00, "not finite"),
            (0x7C00, "not finite"),
        ] {
            let mut map = build(&cfg);
            map.get_mut(&sn).unwrap().data[10..12].copy_from_slice(&bits.to_le_bytes());
            let e = err_of(&map, &cfg);
            assert!(
                e.contains(&sn) && e.contains("[5]") && e.contains(needle),
                "{bits:04x}: {e}"
            );
        }
        // wrong length
        let mut map = build(&cfg);
        put(&mut map, &sn, 1, &[255], f16_vec(&vec![1.0; 255]));
        assert!(err_of(&map, &cfg).contains(&sn));
        // not F16
        let mut map = build(&cfg);
        map.get_mut(&sn).unwrap().info.quant_type = 2;
        assert!(err_of(&map, &cfg).contains("must be F16"));
    }

    #[test]
    fn wrong_down_tail_is_rejected() {
        let cfg = tiny_cfg();
        let sn = awq_sidecar_name(DN);
        let mut map = build(&cfg);
        // tail entry 128 (first padded channel) = 2.0
        map.get_mut(&sn).unwrap().data[256..258].copy_from_slice(&f16b(2.0).to_le_bytes());
        let e = err_of(&map, &cfg);
        assert!(
            e.contains(&sn) && e.contains("[128]") && e.contains("down tail"),
            "{e}"
        );
    }

    #[test]
    fn mixed_qt_in_an_expert_table_and_qt13_are_rejected() {
        let cfg = tiny_cfg();
        let mut map = build(&cfg);
        map.get_mut(GU).unwrap().info.quant_type = 13;
        let e = err_of(&map, &cfg);
        assert!(
            e.contains("layer 1") && e.contains("mixes quant types"),
            "{e}"
        );
        // a stray qt13 tensor outside the expert tables trips the census
        let mut map = build(&cfg);
        put(
            &mut map,
            "model.llm.layers.0.stray.weight",
            13,
            &[1, 256],
            vec![0u8; 136],
        );
        assert!(err_of(&map, &cfg).contains("qt13"));
    }

    #[test]
    fn wrong_shapes_strides_and_router_are_rejected() {
        let cfg = tiny_cfg();
        // legacy-style logical down [e, inter]
        let mut map = build(&cfg);
        map.get_mut(DN).unwrap().info.shape = vec![256, 128];
        let e = err_of(&map, &cfg);
        assert!(e.contains(DN) && e.contains("shape"), "{e}");
        // truncated bytes (wrong row stride)
        let mut map = build(&cfg);
        map.get_mut(DN).unwrap().data.truncate(136 * 255);
        assert!(err_of(&map, &cfg).contains(DN));
        // router must be F16 [n_experts, e]
        let mut map = build(&cfg);
        put(
            &mut map,
            "model.llm.layers.0.mlp.gate.weight",
            44,
            &[2, 256],
            sym_qt44(2, 256, 0),
        );
        assert!(err_of(&map, &cfg).contains("mlp.gate.weight"));
        // sidecar groups of Q / R / gate must be byte-identical
        let mut map = build(&cfg);
        let r = "model.llm.layers.0.attn.wr_du.awq_scale.weight";
        map.get_mut(r).unwrap().data[0..2].copy_from_slice(&f16b(3.0).to_le_bytes());
        let e = err_of(&map, &cfg);
        assert!(e.contains("byte-identical"), "{e}");
        // embed must stay a [vocab, e] qt44
        let mut map = build(&cfg);
        map.get_mut("model.llm.embed.weight")
            .unwrap()
            .info
            .quant_type = 1;
        assert!(err_of(&map, &cfg).contains("model.llm.embed.weight"));
    }

    #[test]
    fn layout_detection_is_typed() {
        let mut meta: serde_json::Value = serde_json::from_str(crate::TEST_METADATA).unwrap();
        assert_eq!(Mq4Layout::from_metadata(&meta).unwrap(), Mq4Layout::Legacy);
        assert_eq!(
            Inkling2Config::from_metadata_json(crate::TEST_METADATA)
                .unwrap()
                .mq4_layout,
            Mq4Layout::Legacy
        );
        meta["hipfire_inkling2_mq4_layout"] = "qt44-padded-awq-v1".into();
        assert_eq!(
            Mq4Layout::from_metadata(&meta).unwrap(),
            Mq4Layout::Qt44PaddedAwqV1
        );
        let cfg = Inkling2Config::from_metadata_json(&meta.to_string()).unwrap();
        assert!(cfg.mq4_layout.is_xts());
        for bad in [
            serde_json::json!("qt44-padded-awq-v2"),
            serde_json::json!(1),
            serde_json::json!(""),
        ] {
            meta["hipfire_inkling2_mq4_layout"] = bad;
            let e = Mq4Layout::from_metadata(&meta).unwrap_err();
            assert!(e.contains("hipfire_inkling2_mq4_layout"), "{e}");
            assert!(Inkling2Config::from_metadata_json(&meta.to_string()).is_err());
        }
    }

    pub(crate) fn weights_with(
        map: HashMap<String, RawTensor>,
        xts: Option<XtsCertificate>,
    ) -> Inkling2Weights {
        Inkling2Weights {
            tensors: map,
            f32_cache: std::cell::RefCell::new(HashMap::new()),
            xts,
        }
    }

    #[test]
    fn xts_f32_divides_columns_by_sidecar_and_slices_down() {
        let cfg = tiny_cfg();
        let map = build(&cfg);
        let cert = certify_xts(&map, &cfg).unwrap();
        let w = weights_with(map.clone(), Some(cert));

        // Dense tensor with a non-unit (2.0) sidecar: deq(W) / 2 per column.
        let name = "model.llm.layers.0.attn.wk_dv.weight";
        let (shape, got) = w.f32(name).unwrap();
        let raw = &map[name];
        let base =
            hipfire_runtime::weight_backend::dequantize_weight_to_f32(44, &raw.data, 128 * 256);
        assert_eq!(shape, vec![128, 256]);
        assert!(base.iter().any(|&v| v != 0.0));
        for (g, b) in got.iter().zip(&base) {
            assert_eq!(g.to_bits(), (b / 2.0).to_bits());
        }

        // Down: physical [256, 256] -> logical [256, 128]; live columns /2, tail dropped.
        let (dshape, dgot) = w.f32(DN).unwrap();
        assert_eq!(dshape, vec![256, 128]);
        let dbase =
            hipfire_runtime::weight_backend::dequantize_weight_to_f32(44, &map[DN].data, 256 * 256);
        for r in 0..256 {
            for c in 0..128 {
                assert_eq!(
                    dgot[r * 128 + c].to_bits(),
                    (dbase[r * 256 + c] / 2.0).to_bits()
                );
            }
        }

        // Embed / unembed / router carry no sidecar: untouched dequant.
        let (_, emb) = w.f32("model.llm.embed.weight").unwrap();
        let ebase = hipfire_runtime::weight_backend::dequantize_weight_to_f32(
            44,
            &map["model.llm.embed.weight"].data,
            4 * 256,
        );
        assert_eq!(emb, ebase);
        let (rs, rg) = w.f32("model.llm.layers.0.mlp.gate.weight").unwrap();
        assert_eq!(rs, vec![2, 256]);
        assert!(rg.iter().all(|&v| v == 0.5));
    }

    #[test]
    fn legacy_weights_f32_is_the_plain_dequant_even_next_to_a_sidecar_name() {
        let cfg = tiny_cfg();
        let map = build(&cfg);
        let w = weights_with(map.clone(), None);
        let name = "model.llm.layers.0.attn.wk_dv.weight";
        let (shape, got) = w.f32(name).unwrap();
        let base = hipfire_runtime::weight_backend::dequantize_weight_to_f32(
            44,
            &map[name].data,
            128 * 256,
        );
        assert_eq!(shape, vec![128, 256]);
        assert_eq!(got, base);
        // legacy expert counting ignores sidecars either way
        assert_eq!(w.expert_tensor_count(), cfg.n_phys * cfg.n_experts * 2);
    }
}
