// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Typed Raon-1-mini weights and the HFQ loader (PLAN §3.2 class map). Owner: slice A.
//!
//! HFQ tensors keep the **original HF names** (`model.layers.N.…`). The
//! single source of truth for names, shapes and storage classes is
//! [`tensor_inventory`]: the loader's pre-flight (`check_inventory`), the
//! upload loop and the unconsumed-tensor census all derive from it, so a
//! tensor can neither be loaded without being declared nor declared without
//! being loaded.
//!
//! Storage classes (PLAN §3.2):
//!
//! | class | stored as | resident as |
//! |---|---|---|
//! | embedding | Q8F16 (qt 3), or qt 1/2/16/44 | raw Q8 table / host-decoded F32 |
//! | projections (attn, dense/shared FFN, lm_head) | Q8F16 (qt 3; every attention projection and lm_head in the shipped recipe, no AWQ sidecar) or qt44 (symmetric MQ4G256V2, dense/shared FFN) when K is a multiple of 256; qt 1/2/16 widen to F32 | [`WeightTensor`] (+ AWQ sidecar on qt44) |
//! | routed experts | symmetric qt44 only; gate rows ++ up rows fused per expert | per-expert `[2*1088, 2560]` and `[2560, 1280]` |
//! | router, mHC selector | qt 1/2/16 | BF16 (qt16 as stored; F16/F32 narrowed with every value verified BF16-representable, else that tensor stays F32) |
//! | norms, conv, A_log, dt_bias, biases | qt 1/2/16 | F32 |
//!
//! The 1088-wide FFN intermediate is *physical* K=1280 for every down
//! projection (routed and shared): the stored tensor is `[2560, 1280]` with
//! the padded natural columns zero before the FWHT; a stored `[2560, 1088]`
//! is refused (it would be silent truncation/padding without a descriptor).
//!
//! Gate||up fusion: a routed expert's `gate_up` is `gate_proj` rows followed
//! by `up_proj` rows (rows are independent in qt44, so the byte concatenation
//! is exact). Dense and shared FFNs keep `gate`/`up` separate.
//!
//! AWQ sidecars follow the repository convention `<stem>.awq_scale.weight`
//! (F16, length = physical K). Layer-level contracts for the MoE:
//! * every routed expert's gate/up sidecar and the shared expert's gate/up
//!   sidecar must be bit-identical (they consume the same rotated activation)
//!   and are exposed once as [`MoeFfnWeights::gate_up_awq`];
//! * per-expert down sidecars (and the shared down sidecar) are exposed as an
//!   `expert_down_awq_ptrs` table; their padded channels `[1088, 1280)` must
//!   be exactly 1.0;
//! * presence is all-or-none per layer.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};

use hipfire_dispatch::families::gemv::WeightRef;
use hipfire_dispatch::families::moe::RoutedExpertWeights;
use hipfire_dispatch::pipeline::{
    grouped_sym_tier_admits, ExpertBindingCache, ExpertMetadata, ExpertResource, ExpertResources,
    ExpertTable, PrefillActTier,
};
use hipfire_dispatch::types::dtype_rotation_plan;
use hipfire_runtime::hfq::{HfqFile, HfqTensorInfo};
use hipfire_runtime::llama::{f16_to_f32, f32_to_f16, EmbeddingFormat, WeightTensor};
use hipfire_runtime::weight_backend::{dequant_weight_raw, load_embedding, MemoryTarget};
use rdna_compute::{DType, Gpu, GpuTensor};

use crate::config::{FfnKind, LayerKind, RaonConfig, MHC_STREAMS, QT44_GROUP};

// ───────────────────────── quant-type constants ─────────────────────────

/// HFQ quant-type bytes this loader understands.
const QT_F16: u8 = 1;
const QT_F32: u8 = 2;
const QT_Q8F16: u8 = 3;
const QT_MQ4V2: u8 = 44;
const QT_BF16: u8 = 16;

/// mHC site width: selector rows `n + n + (n-1)^2`, input width `n * hidden`.
pub const MHC_CONTROLS: usize = 2 * MHC_STREAMS + (MHC_STREAMS - 1) * (MHC_STREAMS - 1);
/// Layout of [`MhcSiteWeights::params`] (F32).
pub const MHC_PARAM_ALPHA_PRE: usize = 0;
pub const MHC_PARAM_ALPHA_POST: usize = 1;
pub const MHC_PARAM_ALPHA_Z: usize = 2;
pub const MHC_PARAM_ETA: usize = 3;
pub const MHC_PARAM_B_PRE: usize = 4;
pub const MHC_PARAM_B_POST: usize = MHC_PARAM_B_PRE + MHC_STREAMS;
pub const MHC_PARAM_B_Z: usize = MHC_PARAM_B_POST + MHC_STREAMS;
pub const MHC_PARAM_LEN: usize = MHC_PARAM_B_Z + (MHC_STREAMS - 1) * (MHC_STREAMS - 1);

// ───────────────────────── typed weights ─────────────────────────

/// One mHC site (`self_attn_raon_mhc` / `mlp_raon_mhc`).
pub struct MhcSiteWeights {
    /// BF16 `[17, n*hidden]` (F32 only when a value is not BF16-representable):
    /// rows `0..4` = `phi_pre`, `4..8` = `phi_post`, `8..17` = `res.phi_z`
    /// (the BF16 checkpoint values, exact).
    pub selector: WeightTensor,
    /// F32 `[MHC_PARAM_LEN]`, layout per the `MHC_PARAM_*` constants:
    /// `alpha_pre, alpha_post, alpha_z, eta, b_pre[4], b_post[4], b_z[9]`.
    pub params: GpuTensor,
    // Host copies of `params` (the oracle and graph builders read these).
    pub alpha_pre: f32,
    pub alpha_post: f32,
    pub alpha_z: f32,
    pub eta: f32,
    pub b_pre: [f32; MHC_STREAMS],
    pub b_post: [f32; MHC_STREAMS],
    pub b_z: [f32; (MHC_STREAMS - 1) * (MHC_STREAMS - 1)],
}

/// RDA token mixer (`self_attn.*` of an RDA layer).
pub struct RdaWeights {
    pub q_proj: WeightTensor,
    pub k_proj: WeightTensor,
    pub v_proj: WeightTensor,
    /// Output gate projection `[5120, 2560]`.
    pub g_proj: WeightTensor,
    pub o_proj: WeightTensor,
    /// Decay bottleneck: `f_a [128, 2560]`, `f_b [5120, 128]`.
    pub f_a_proj: WeightTensor,
    pub f_b_proj: WeightTensor,
    /// Erase gate logits `b_proj [40, 2560]`, write-gate delta `rda_w_proj [40, 2560]`.
    pub b_proj: WeightTensor,
    pub rda_w_proj: WeightTensor,
    /// Depthwise causal conv taps, F32 `[5120, 4]` (oldest→newest tap order).
    pub q_conv: GpuTensor,
    pub k_conv: GpuTensor,
    pub v_conv: GpuTensor,
    /// Raw `A_log` F32 `[40]` (the decay uses `exp(A_log)` inside the sigmoid).
    pub a_log: GpuTensor,
    /// F32 `[5120]`.
    pub dt_bias: GpuTensor,
    /// Shared per-head output RMS gamma, F32 `[128]`.
    pub o_norm: GpuTensor,
}

/// NoPE MLA token mixer (`self_attn.*` of an MLA layer).
pub struct MlaWeights {
    pub q_a_proj: WeightTensor,
    /// F32 `[576]`.
    pub q_a_norm: GpuTensor,
    /// `[7680, 576]`; Q8F16 (K=576 is not a qt44 group multiple).
    pub q_b_proj: WeightTensor,
    pub kv_a_proj: WeightTensor,
    /// F32 `[512]`.
    pub kv_a_norm: GpuTensor,
    pub kv_b_proj: WeightTensor,
    pub g_proj: WeightTensor,
    pub o_proj: WeightTensor,
}

/// Gated SiTU FFN: dense layer-0 MLP (12032) or the shared expert (1088, down K=1280).
///
/// `gate`/`up`/`down` are what prefill reads. A projection stored BF16
/// (source-exact) is resident as its exact F16 image there and has a load-time
/// Q8_0 copy in `decode` (`[gate, up, down]`) that single-row decode reads
/// instead (the Flash-Next shared-expert scheme); `None` = decode reads the
/// resident tensor (qt44 / Q8F16 artifacts).
pub struct GatedFfnWeights {
    pub gate: WeightTensor,
    pub up: WeightTensor,
    pub down: WeightTensor,
    pub decode: [Option<WeightTensor>; 3],
    /// Every qt44 weight's group headers are symmetric (`zp == -8 * sc`), verified on the device at
    /// load (gfx1201 only); admits the grouped symmetric quantized-activation cores for this FFN.
    pub iu4_symmetric: bool,
}

impl GatedFfnWeights {
    /// `(gate, up, down)` for a call over `rows` rows: the decode copies at
    /// `rows == 1`, the resident tensors otherwise.
    pub fn projections(&self, rows: usize) -> (&WeightTensor, &WeightTensor, &WeightTensor) {
        fn pick<'a>(
            rows: usize,
            resident: &'a WeightTensor,
            copy: &'a Option<WeightTensor>,
        ) -> &'a WeightTensor {
            if rows == 1 {
                copy.as_ref().unwrap_or(resident)
            } else {
                resident
            }
        }
        (
            pick(rows, &self.gate, &self.decode[0]),
            pick(rows, &self.up, &self.decode[1]),
            pick(rows, &self.down, &self.decode[2]),
        )
    }
}

/// One routed expert, rows fused: `gate_up` is gate rows `0..1088` then up
/// rows `1088..2176`.
pub struct ExpertWeights {
    pub gate_up: WeightTensor,
    pub down: WeightTensor,
}

/// Sigmoid top-k MoE FFN.
pub struct MoeFfnWeights {
    /// BF16 `[128, 2560]` (F32 only when a value is not BF16-representable).
    pub router: WeightTensor,
    /// F32 `[128]`, selection-only bias.
    pub e_bias: GpuTensor,
    pub shared: GatedFfnWeights,
    pub experts: Vec<ExpertWeights>,
    /// `[2 * n_experts]` F32 slots = `n_experts` device `u64` pointers to each
    /// expert's `gate_up` / `down` allocation (Qwen35/LFM layout).
    pub expert_gate_up_ptrs: GpuTensor,
    pub expert_down_ptrs: GpuTensor,
    /// Layer-wide gate/up AWQ scale, F32 `[2560]` (identical for every routed
    /// expert and the shared expert), when the artifact carries sidecars.
    pub gate_up_awq: Option<GpuTensor>,
    /// Pointer table to per-expert down AWQ scales (F32 `[1280]` each).
    pub expert_down_awq_ptrs: Option<GpuTensor>,
    /// Owners of the per-expert down AWQ tensors behind `expert_down_awq_ptrs`.
    pub expert_down_awq: Vec<GpuTensor>,
    /// Load-time expert identity for the shared sealed-MoE executor.
    pub table: ExpertTable,
    pub cache: ExpertBindingCache,
    /// Every routed gate/up and down header is symmetric (`zp == -8 * sc`), verified on the device
    /// at load on archs that admit a quantized grouped tier (gfx1201, gfx1100, gfx1151); admits the grouped
    /// W4A8 / W4A4 expert core.
    pub iu4_symmetric: bool,
}

pub enum RaonMixer {
    Rda(RdaWeights),
    Mla(MlaWeights),
}

pub enum RaonFfn {
    Dense(GatedFfnWeights),
    Moe(MoeFfnWeights),
}

pub struct RaonLayerWeights {
    /// `input_layernorm`, F32 `[2560]`.
    pub input_norm: GpuTensor,
    /// `post_attention_layernorm`, F32 `[2560]`.
    pub post_attn_norm: GpuTensor,
    pub attn_mhc: MhcSiteWeights,
    pub ffn_mhc: MhcSiteWeights,
    pub mixer: RaonMixer,
    pub ffn: RaonFfn,
}

/// Immutable device weights of one Raon-1-mini model.
pub struct RaonWeights {
    /// `model.embed_tokens.weight` in its on-device format (see `embed_format`).
    pub embed: GpuTensor,
    pub embed_format: EmbeddingFormat,
    /// `model.norm.weight`, F32 `[2560]`.
    pub final_norm: GpuTensor,
    /// Untied `lm_head.weight` `[vocab, 2560]`.
    pub lm_head: WeightTensor,
    pub layers: Vec<RaonLayerWeights>,
}

// ───────────────────────── free ─────────────────────────

fn free_t(gpu: &mut Gpu, t: GpuTensor) {
    let _ = gpu.free_tensor(t);
}

impl MhcSiteWeights {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        self.selector.free_all(gpu);
        free_t(gpu, self.params);
    }
}

impl RdaWeights {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        for w in [
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.g_proj,
            self.o_proj,
            self.f_a_proj,
            self.f_b_proj,
            self.b_proj,
            self.rda_w_proj,
        ] {
            w.free_all(gpu);
        }
        for t in [
            self.q_conv,
            self.k_conv,
            self.v_conv,
            self.a_log,
            self.dt_bias,
            self.o_norm,
        ] {
            free_t(gpu, t);
        }
    }
}

impl MlaWeights {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        for w in [
            self.q_a_proj,
            self.q_b_proj,
            self.kv_a_proj,
            self.kv_b_proj,
            self.g_proj,
            self.o_proj,
        ] {
            w.free_all(gpu);
        }
        free_t(gpu, self.q_a_norm);
        free_t(gpu, self.kv_a_norm);
    }
}

impl GatedFfnWeights {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        self.gate.free_all(gpu);
        self.up.free_all(gpu);
        self.down.free_all(gpu);
        for d in self.decode.into_iter().flatten() {
            d.free_all(gpu);
        }
    }
}

impl ExpertWeights {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        self.gate_up.free_all(gpu);
        self.down.free_all(gpu);
    }
}

impl MoeFfnWeights {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let MoeFfnWeights {
            router,
            e_bias,
            shared,
            experts,
            expert_gate_up_ptrs,
            expert_down_ptrs,
            gate_up_awq,
            expert_down_awq_ptrs,
            expert_down_awq,
            table: _,
            cache: _,
            iu4_symmetric: _,
        } = self;
        router.free_all(gpu);
        free_t(gpu, e_bias);
        shared.free_gpu(gpu);
        for e in experts {
            e.free_gpu(gpu);
        }
        free_t(gpu, expert_gate_up_ptrs);
        free_t(gpu, expert_down_ptrs);
        if let Some(t) = gate_up_awq {
            free_t(gpu, t);
        }
        if let Some(t) = expert_down_awq_ptrs {
            free_t(gpu, t);
        }
        for t in expert_down_awq {
            free_t(gpu, t);
        }
    }
}

impl RaonMixer {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        match self {
            RaonMixer::Rda(w) => w.free_gpu(gpu),
            RaonMixer::Mla(w) => w.free_gpu(gpu),
        }
    }
}

impl RaonFfn {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        match self {
            RaonFfn::Dense(w) => w.free_gpu(gpu),
            RaonFfn::Moe(w) => w.free_gpu(gpu),
        }
    }
}

impl RaonLayerWeights {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        free_t(gpu, self.input_norm);
        free_t(gpu, self.post_attn_norm);
        self.attn_mhc.free_gpu(gpu);
        self.ffn_mhc.free_gpu(gpu);
        self.mixer.free_gpu(gpu);
        self.ffn.free_gpu(gpu);
    }
}

impl RaonWeights {
    /// Return every device buffer this model owns.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        // Layers first (the many small allocations), then the big tables.
        for layer in self.layers {
            layer.free_gpu(gpu);
        }
        self.lm_head.free_all(gpu);
        free_t(gpu, self.final_norm);
        free_t(gpu, self.embed);
    }
}

/// Routed experts as the sealed executor reads them.
struct ResidentExperts<'a>(&'a [ExpertWeights]);

impl RoutedExpertWeights for ResidentExperts<'_> {
    fn len(&self) -> usize {
        self.0.len()
    }
    fn get(&self, expert: usize) -> Option<(WeightRef<'_>, WeightRef<'_>)> {
        self.0
            .get(expert)
            .map(|e| (e.gate_up.dispatch_ref(), e.down.dispatch_ref()))
    }
}

impl RoutedExpertWeights for MoeFfnWeights {
    fn len(&self) -> usize {
        self.experts.len()
    }
    fn get(&self, expert: usize) -> Option<(WeightRef<'_>, WeightRef<'_>)> {
        self.experts
            .get(expert)
            .map(|e| (e.gate_up.dispatch_ref(), e.down.dispatch_ref()))
    }
}

// ───────────────────────── inventory (name map) ─────────────────────────

/// Storage class of a declared tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorClass {
    /// `model.embed_tokens.weight`.
    Embedding,
    /// Matrix projection: qt44 when K is a group multiple, Q8F16 otherwise;
    /// qt 1/2/16 widen to F32. May carry an AWQ sidecar.
    Proj,
    /// Router matrix: widened to a device F32 `WeightTensor`.
    Router,
    /// Small F32 vector kept on device (norms, conv, A_log, dt_bias, e_bias).
    F32Vec,
    /// Small tensor staged on the host (mHC selector rows, scalars, biases).
    HostF32,
    /// Routed expert part: qt44 only; fused/uploaded per layer.
    ExpertGate,
    ExpertUp,
    ExpertDown,
}

/// One declared tensor: original HF name, storage class, stored shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorSpec {
    pub name: String,
    pub class: TensorClass,
    /// Stored (physical) shape. Empty = 0-d scalar (one element; `[1]` also accepted).
    pub shape: Vec<usize>,
}

impl TensorSpec {
    fn new(name: impl Into<String>, class: TensorClass, shape: &[usize]) -> Self {
        Self {
            name: name.into(),
            class,
            shape: shape.to_vec(),
        }
    }
    fn numel(&self) -> usize {
        self.shape.iter().product()
    }
    fn is_matrix_proj(&self) -> bool {
        matches!(
            self.class,
            TensorClass::Proj
                | TensorClass::ExpertGate
                | TensorClass::ExpertUp
                | TensorClass::ExpertDown
        )
    }
}

/// AWQ sidecar name of a `<stem>.weight` tensor.
pub fn awq_sidecar_name(weight_name: &str) -> String {
    match weight_name.strip_suffix(".weight") {
        Some(stem) => format!("{stem}.awq_scale.weight"),
        None => format!("{weight_name}.awq_scale.weight"),
    }
}

fn mhc_specs(out: &mut Vec<TensorSpec>, p: &str, cfg: &RaonConfig) {
    let n = cfg.mhc_streams;
    let w = n * cfg.hidden;
    let m = (n - 1) * (n - 1);
    out.push(TensorSpec::new(
        format!("{p}.phi_pre"),
        TensorClass::HostF32,
        &[n, w],
    ));
    out.push(TensorSpec::new(
        format!("{p}.phi_post"),
        TensorClass::HostF32,
        &[n, w],
    ));
    out.push(TensorSpec::new(
        format!("{p}.res.phi_z"),
        TensorClass::HostF32,
        &[m, w],
    ));
    out.push(TensorSpec::new(
        format!("{p}.alpha_pre"),
        TensorClass::HostF32,
        &[],
    ));
    out.push(TensorSpec::new(
        format!("{p}.alpha_post"),
        TensorClass::HostF32,
        &[],
    ));
    out.push(TensorSpec::new(
        format!("{p}.res.alpha_z"),
        TensorClass::HostF32,
        &[],
    ));
    out.push(TensorSpec::new(
        format!("{p}.res.eta"),
        TensorClass::HostF32,
        &[],
    ));
    out.push(TensorSpec::new(
        format!("{p}.b_pre"),
        TensorClass::HostF32,
        &[n],
    ));
    out.push(TensorSpec::new(
        format!("{p}.b_post"),
        TensorClass::HostF32,
        &[n],
    ));
    out.push(TensorSpec::new(
        format!("{p}.res.b_z"),
        TensorClass::HostF32,
        &[m],
    ));
}

/// Every tensor the artifact must carry for `cfg`, with original HF names
/// and stored shapes. Order: globals, then layers ascending.
pub fn tensor_inventory(cfg: &RaonConfig) -> Vec<TensorSpec> {
    use TensorClass::*;
    let c = cfg.hidden;
    let p = cfg.rda_proj;
    let mut v = Vec::new();
    v.push(TensorSpec::new(
        "model.embed_tokens.weight",
        Embedding,
        &[cfg.vocab, c],
    ));
    v.push(TensorSpec::new("model.norm.weight", F32Vec, &[c]));
    v.push(TensorSpec::new("lm_head.weight", Proj, &[cfg.vocab, c]));
    for l in 0..cfg.n_layers {
        let lp = format!("model.layers.{l}");
        v.push(TensorSpec::new(
            format!("{lp}.input_layernorm.weight"),
            F32Vec,
            &[c],
        ));
        v.push(TensorSpec::new(
            format!("{lp}.post_attention_layernorm.weight"),
            F32Vec,
            &[c],
        ));
        mhc_specs(&mut v, &format!("{lp}.self_attn_raon_mhc"), cfg);
        mhc_specs(&mut v, &format!("{lp}.mlp_raon_mhc"), cfg);
        let a = format!("{lp}.self_attn");
        match cfg.layer_kinds[l] {
            LayerKind::Rda => {
                for n in ["q_proj", "k_proj", "v_proj", "g_proj"] {
                    v.push(TensorSpec::new(format!("{a}.{n}.weight"), Proj, &[p, c]));
                }
                v.push(TensorSpec::new(format!("{a}.o_proj.weight"), Proj, &[c, p]));
                v.push(TensorSpec::new(
                    format!("{a}.f_a_proj.weight"),
                    Proj,
                    &[cfg.rda_decay_rank, c],
                ));
                v.push(TensorSpec::new(
                    format!("{a}.f_b_proj.weight"),
                    Proj,
                    &[p, cfg.rda_decay_rank],
                ));
                v.push(TensorSpec::new(
                    format!("{a}.b_proj.weight"),
                    Proj,
                    &[cfg.n_heads, c],
                ));
                v.push(TensorSpec::new(
                    format!("{a}.rda_w_proj.weight"),
                    Proj,
                    &[cfg.n_heads, c],
                ));
                for n in ["q_conv1d", "k_conv1d", "v_conv1d"] {
                    v.push(TensorSpec::new(
                        format!("{a}.{n}.weight"),
                        F32Vec,
                        &[p, cfg.rda_conv_kernel],
                    ));
                }
                v.push(TensorSpec::new(
                    format!("{a}.A_log"),
                    F32Vec,
                    &[cfg.n_heads],
                ));
                v.push(TensorSpec::new(format!("{a}.dt_bias"), F32Vec, &[p]));
                v.push(TensorSpec::new(
                    format!("{a}.o_norm.weight"),
                    F32Vec,
                    &[cfg.head_dim],
                ));
            }
            LayerKind::Mla => {
                v.push(TensorSpec::new(
                    format!("{a}.q_a_proj.weight"),
                    Proj,
                    &[cfg.q_lora, c],
                ));
                v.push(TensorSpec::new(
                    format!("{a}.q_a_layernorm.weight"),
                    F32Vec,
                    &[cfg.q_lora],
                ));
                v.push(TensorSpec::new(
                    format!("{a}.q_b_proj.weight"),
                    Proj,
                    &[cfg.n_heads * cfg.mla_qk_head(), cfg.q_lora],
                ));
                v.push(TensorSpec::new(
                    format!("{a}.kv_a_proj_with_mqa.weight"),
                    Proj,
                    &[cfg.kv_lora + cfg.qk_rope, c],
                ));
                v.push(TensorSpec::new(
                    format!("{a}.kv_a_layernorm.weight"),
                    F32Vec,
                    &[cfg.kv_lora],
                ));
                v.push(TensorSpec::new(
                    format!("{a}.kv_b_proj.weight"),
                    Proj,
                    &[cfg.n_heads * (cfg.qk_nope + cfg.v_head), cfg.kv_lora],
                ));
                v.push(TensorSpec::new(
                    format!("{a}.g_proj.weight"),
                    Proj,
                    &[cfg.n_heads * cfg.v_head, c],
                ));
                v.push(TensorSpec::new(
                    format!("{a}.o_proj.weight"),
                    Proj,
                    &[c, cfg.n_heads * cfg.v_head],
                ));
            }
        }
        match cfg.ffn_kinds[l] {
            FfnKind::Dense => {
                let m = format!("{lp}.mlp");
                v.push(TensorSpec::new(
                    format!("{m}.gate_proj.weight"),
                    Proj,
                    &[cfg.dense_inter, c],
                ));
                v.push(TensorSpec::new(
                    format!("{m}.up_proj.weight"),
                    Proj,
                    &[cfg.dense_inter, c],
                ));
                v.push(TensorSpec::new(
                    format!("{m}.down_proj.weight"),
                    Proj,
                    &[c, cfg.dense_inter],
                ));
            }
            FfnKind::Moe => {
                let m = format!("{lp}.moe");
                v.push(TensorSpec::new(
                    format!("{m}.gate.weight"),
                    Router,
                    &[cfg.n_experts, c],
                ));
                v.push(TensorSpec::new(
                    format!("{m}.gate.e_score_correction_bias"),
                    F32Vec,
                    &[cfg.n_experts],
                ));
                let si = cfg.shared_inter();
                let sk = cfg.shared_down_k();
                v.push(TensorSpec::new(
                    format!("{m}.shared_experts.gate_proj.weight"),
                    Proj,
                    &[si, c],
                ));
                v.push(TensorSpec::new(
                    format!("{m}.shared_experts.up_proj.weight"),
                    Proj,
                    &[si, c],
                ));
                v.push(TensorSpec::new(
                    format!("{m}.shared_experts.down_proj.weight"),
                    Proj,
                    &[c, sk],
                ));
                for e in 0..cfg.n_experts {
                    let ep = format!("{m}.experts.{e}");
                    v.push(TensorSpec::new(
                        format!("{ep}.gate_proj.weight"),
                        ExpertGate,
                        &[cfg.moe_inter, c],
                    ));
                    v.push(TensorSpec::new(
                        format!("{ep}.up_proj.weight"),
                        ExpertUp,
                        &[cfg.moe_inter, c],
                    ));
                    v.push(TensorSpec::new(
                        format!("{ep}.down_proj.weight"),
                        ExpertDown,
                        &[c, cfg.moe_down_k],
                    ));
                }
            }
        }
    }
    v
}

// ───────────────────────── pre-flight ─────────────────────────

/// One tensor-index row as the pre-flight sees it (decoupled from `HfqFile` so
/// the checks unit-test without a container).
#[derive(Clone, Debug)]
pub struct IndexEntry {
    pub name: String,
    pub quant_type: u8,
    pub shape: Vec<usize>,
    pub data_size: usize,
}

impl IndexEntry {
    fn from_info(i: &HfqTensorInfo) -> Self {
        Self {
            name: i.name.clone(),
            quant_type: i.quant_type,
            shape: i.shape.iter().map(|&d| d as usize).collect(),
            data_size: i.data_size,
        }
    }
}

/// Exact payload bytes for `[m, k]` at quant type `qt`, or `None` when the
/// (qt, k) pair is not a valid layout.
fn payload_bytes(qt: u8, m: usize, k: usize) -> Option<usize> {
    match qt {
        QT_MQ4V2 if k % QT44_GROUP == 0 => m.checked_mul(k / QT44_GROUP)?.checked_mul(136),
        QT_Q8F16 if k % 32 == 0 => m.checked_mul(k / 32)?.checked_mul(34),
        QT_F16 | QT_BF16 => m.checked_mul(k)?.checked_mul(2),
        QT_F32 => m.checked_mul(k)?.checked_mul(4),
        _ => None,
    }
}

fn allowed_qts(class: TensorClass) -> &'static [u8] {
    match class {
        TensorClass::Embedding => &[QT_Q8F16, QT_BF16, QT_F16, QT_F32, QT_MQ4V2],
        TensorClass::Proj => &[QT_MQ4V2, QT_Q8F16, QT_BF16, QT_F16, QT_F32],
        TensorClass::ExpertGate | TensorClass::ExpertUp | TensorClass::ExpertDown => &[QT_MQ4V2],
        TensorClass::Router | TensorClass::F32Vec | TensorClass::HostF32 => {
            &[QT_BF16, QT_F16, QT_F32]
        }
    }
}

fn qt_name(qt: u8) -> &'static str {
    match qt {
        QT_F16 => "F16",
        QT_F32 => "F32",
        QT_Q8F16 => "Q8F16",
        QT_BF16 => "BF16",
        QT_MQ4V2 => "qt44/MQ4G256V2",
        _ => "other",
    }
}

fn list_cap(items: &[String]) -> String {
    const CAP: usize = 16;
    if items.len() <= CAP {
        items.join(", ")
    } else {
        format!(
            "{}, … (+{} more)",
            items[..CAP].join(", "),
            items.len() - CAP
        )
    }
}

/// Pre-flight against the artifact index: every declared tensor present with
/// the declared shape, an admissible quant type and an exactly sized payload;
/// no AWQ sidecar on a dtype that cannot use one; and no tensor in the
/// artifact that the inventory does not declare (unconsumed). All problems are
/// collected and reported together.
pub fn check_inventory(specs: &[TensorSpec], index: &[IndexEntry]) -> Result<(), String> {
    let by_name: HashMap<&str, &IndexEntry> = index.iter().map(|e| (e.name.as_str(), e)).collect();
    let declared: HashSet<&str> = specs.iter().map(|s| s.name.as_str()).collect();
    let mut missing = Vec::new();
    let mut bad = Vec::new();
    let mut sidecar_names: HashSet<String> = HashSet::new();
    for s in specs {
        let Some(e) = by_name.get(s.name.as_str()) else {
            missing.push(s.name.clone());
            continue;
        };
        let shape_ok = if s.shape.is_empty() {
            e.shape.iter().product::<usize>() == 1
        } else {
            e.shape == s.shape
        };
        if !shape_ok {
            let hint = if s.is_matrix_proj()
                && s.shape.len() == 2
                && e.shape.len() == 2
                && e.shape[0] == s.shape[0]
                && e.shape[1] < s.shape[1]
            {
                " (unpadded K: the 1088-wide FFN intermediate must be stored with physical K=1280)"
            } else {
                ""
            };
            bad.push(format!(
                "{}: shape {:?}, expected {:?}{hint}",
                s.name, e.shape, s.shape
            ));
            continue;
        }
        if !allowed_qts(s.class).contains(&e.quant_type) {
            bad.push(format!(
                "{}: quant type {} ({}) not admitted for {:?} (allowed: {})",
                s.name,
                e.quant_type,
                qt_name(e.quant_type),
                s.class,
                allowed_qts(s.class)
                    .iter()
                    .map(|q| qt_name(*q))
                    .collect::<Vec<_>>()
                    .join("/")
            ));
            continue;
        }
        let (rows, cols) = match s.shape.as_slice() {
            [] => (1, 1),
            [n] => (1, *n),
            [m, k] => (*m, *k),
            _ => unreachable!("inventory shapes are rank <= 2"),
        };
        match payload_bytes(e.quant_type, rows, cols) {
            Some(want) if want == e.data_size => {}
            Some(want) => bad.push(format!(
                "{}: payload {} B, expected {want} B for {} {rows}x{cols}",
                s.name,
                e.data_size,
                qt_name(e.quant_type)
            )),
            None => bad.push(format!(
                "{}: {} cannot encode K={cols} (qt44 needs K%256==0, Q8F16 needs K%32==0)",
                s.name,
                qt_name(e.quant_type)
            )),
        }
        if s.is_matrix_proj() {
            let sc = awq_sidecar_name(&s.name);
            if let Some(side) = by_name.get(sc.as_str()) {
                sidecar_names.insert(sc.clone());
                if e.quant_type != QT_MQ4V2 {
                    bad.push(format!(
                        "{sc}: AWQ sidecar on a {} weight (only qt44 consumes AWQ scales)",
                        qt_name(e.quant_type)
                    ));
                } else if side.quant_type != QT_F16
                    || side.shape != [cols]
                    || side.data_size != 2 * cols
                {
                    bad.push(format!(
                        "{sc}: sidecar must be F16 [{cols}] (got qt {} shape {:?}, {} B)",
                        side.quant_type, side.shape, side.data_size
                    ));
                }
            }
        }
    }
    let mut unconsumed: Vec<String> = index
        .iter()
        .filter(|e| !declared.contains(e.name.as_str()) && !sidecar_names.contains(&e.name))
        .map(|e| e.name.clone())
        .collect();
    unconsumed.sort();
    let mut errs = Vec::new();
    if !missing.is_empty() {
        errs.push(format!(
            "missing {} tensor(s): {}",
            missing.len(),
            list_cap(&missing)
        ));
    }
    if !bad.is_empty() {
        errs.push(format!(
            "{} malformed tensor(s): {}",
            bad.len(),
            list_cap(&bad)
        ));
    }
    if !unconsumed.is_empty() {
        errs.push(format!(
            "{} unconsumed tensor(s) (not part of raon1_mini for this config; a raw MXFP8 \
             `weight_scale_inv` here means the artifact was not produced by `hipfire quantize`): {}",
            unconsumed.len(),
            list_cap(&unconsumed)
        ));
    }
    if errs.is_empty() {
        Ok(())
    } else {
        Err(format!("raon1_mini weights: {}", errs.join("; ")))
    }
}

// ───────────────────────── host decode helpers ─────────────────────────

fn widen_f32(qt: u8, data: &[u8], n: usize, name: &str) -> Result<Vec<f32>, String> {
    let out: Vec<f32> = match qt {
        QT_F16 => data
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        QT_BF16 => data
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect(),
        QT_F32 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        other => {
            return Err(format!(
                "raon1_mini weights: {name}: cannot widen quant type {other} to F32"
            ))
        }
    };
    if out.len() != n {
        return Err(format!(
            "raon1_mini weights: {name}: {} elements, expected {n}",
            out.len()
        ));
    }
    Ok(out)
}

fn read_bytes<R>(
    hfq: &HfqFile,
    name: &str,
    f: impl FnOnce(&HfqTensorInfo, &[u8]) -> Result<R, String>,
) -> Result<R, String> {
    if let Some((info, data)) = hfq.tensor_data(name) {
        return f(info, data);
    }
    let (info, data) = hfq
        .tensor_data_vec(name)
        .ok_or_else(|| format!("raon1_mini weights: tensor {name} unreadable"))?;
    f(info, &data)
}

/// Host copy of an AWQ sidecar (`None` when absent): F16 `[k]`, finite and positive.
fn read_awq_host(hfq: &HfqFile, weight_name: &str, k: usize) -> Result<Option<Vec<f32>>, String> {
    let sc = awq_sidecar_name(weight_name);
    if hfq.find_tensor_info(&sc).is_none() {
        return Ok(None);
    }
    read_bytes(hfq, &sc, |info, data| {
        if info.quant_type != QT_F16 || info.shape.len() != 1 || info.shape[0] as usize != k {
            return Err(format!(
                "raon1_mini weights: {sc}: sidecar must be F16 [{k}], got qt {} shape {:?}",
                info.quant_type, info.shape
            ));
        }
        let v = widen_f32(QT_F16, data, k, &sc)?;
        if let Some(i) = v.iter().position(|x| !x.is_finite() || *x <= 0.0) {
            return Err(format!(
                "raon1_mini weights: {sc}: scale[{i}] = {} is not finite-positive",
                v[i]
            ));
        }
        Ok(Some(v))
    })
}

fn upload_f32_named(
    gpu: &mut Gpu,
    data: &[f32],
    shape: &[usize],
    name: &str,
) -> Result<GpuTensor, String> {
    gpu.upload_f32(data, shape)
        .map_err(|e| format!("raon1_mini weights: upload {name}: {e:?}"))
}

// ───────────────────────── loader ─────────────────────────

enum Loaded {
    Vec(GpuTensor),
    Mat(WeightTensor),
    /// A projection with a separate decode copy (BF16 FFN: F16 image + Q8_0).
    MatDecode(WeightTensor, WeightTensor),
    Host(Vec<f32>),
    Embed(GpuTensor, EmbeddingFormat),
}

/// Every tensor uploaded so far, keyed by HF name. The pool is the sole owner
/// until a group is assembled, so an error anywhere frees everything.
#[derive(Default)]
struct Pool {
    items: HashMap<String, Loaded>,
}

impl Pool {
    fn free(self, gpu: &mut Gpu) {
        for (_, item) in self.items {
            match item {
                Loaded::Vec(t) | Loaded::Embed(t, _) => free_t(gpu, t),
                Loaded::Mat(w) => w.free_all(gpu),
                Loaded::MatDecode(w, d) => {
                    w.free_all(gpu);
                    d.free_all(gpu);
                }
                Loaded::Host(_) => {}
            }
        }
    }
    fn has(&self, name: &str) -> bool {
        self.items.contains_key(name)
    }
    // The takes below are only called after `require` succeeded for the whole
    // group, so the kind checks cannot fail on an inventory-consistent pool.
    fn take_vec(&mut self, name: &str) -> GpuTensor {
        match self.items.remove(name) {
            Some(Loaded::Vec(t)) => t,
            _ => unreachable!("pool entry {name} verified as device vector"),
        }
    }
    fn take_mat(&mut self, name: &str) -> WeightTensor {
        match self.items.remove(name) {
            Some(Loaded::Mat(w)) => w,
            _ => unreachable!("pool entry {name} verified as weight matrix"),
        }
    }
    /// An always-active FFN projection and its decode copy, when it has one.
    fn take_ffn_mat(&mut self, name: &str) -> (WeightTensor, Option<WeightTensor>) {
        match self.items.remove(name) {
            Some(Loaded::Mat(w)) => (w, None),
            Some(Loaded::MatDecode(w, d)) => (w, Some(d)),
            _ => unreachable!("pool entry {name} verified as weight matrix"),
        }
    }
    fn take_host(&mut self, name: &str) -> Vec<f32> {
        match self.items.remove(name) {
            Some(Loaded::Host(v)) => v,
            _ => unreachable!("pool entry {name} verified as host vector"),
        }
    }
    fn kind_ok(&self, spec: &TensorSpec) -> bool {
        match (self.items.get(&spec.name), spec.class) {
            (
                Some(Loaded::Mat(_) | Loaded::MatDecode(..)),
                TensorClass::Proj | TensorClass::Router,
            ) => true,
            (Some(Loaded::Vec(_)), TensorClass::F32Vec) => true,
            (Some(Loaded::Host(_)), TensorClass::HostF32) => true,
            (Some(Loaded::Embed(..)), TensorClass::Embedding) => true,
            _ => false,
        }
    }
}

/// Counters of the router / mHC-selector residency (reported once per load).
static ROWS_BF16_STORED: AtomicUsize = AtomicUsize::new(0);
static ROWS_BF16_NARROWED: AtomicUsize = AtomicUsize::new(0);
static ROWS_F32_KEPT: AtomicUsize = AtomicUsize::new(0);
/// Device bytes the router / selector tensors occupy, and what F32 copies would.
static ROWS_BYTES_RESIDENT: AtomicUsize = AtomicUsize::new(0);
static ROWS_BYTES_AS_F32: AtomicUsize = AtomicUsize::new(0);

/// Exact BF16 image of `data`: `Ok(bits)` when every value is exactly
/// BF16-representable (low 16 bits of the F32 pattern are zero), else
/// `Err((first inexact index, inexact count))`.
fn bf16_exact_image(data: &[f32]) -> Result<Vec<u16>, (usize, usize)> {
    let inexact = data.iter().filter(|v| v.to_bits() & 0xffff != 0).count();
    if inexact != 0 {
        let first = data
            .iter()
            .position(|v| v.to_bits() & 0xffff != 0)
            .unwrap_or(0);
        return Err((first, inexact));
    }
    Ok(data.iter().map(|v| (v.to_bits() >> 16) as u16).collect())
}

fn rows_weight_tensor(buf: GpuTensor, dtype: DType, m: usize, k: usize) -> WeightTensor {
    WeightTensor {
        buf,
        gpu_dtype: dtype,
        m,
        k,
        row_stride: 0,
        paro: None,
        awq_scale: None,
        lloyd_lut_e4m3: None,
        lloyd_lut_f16: None,
        lloyd_lut_c16: None,
    }
}

/// F32 `[m, k]` projection weight (qt 1/2/16 projections that no codec covers).
fn f32_weight(
    gpu: &mut Gpu,
    data: &[f32],
    m: usize,
    k: usize,
    name: &str,
) -> Result<WeightTensor, String> {
    let buf = upload_f32_named(gpu, data, &[m, k], name)?;
    Ok(rows_weight_tensor(buf, DType::F32, m, k))
}

/// Uploads little-endian BF16 words as a device BF16 `[m, k]` weight (the
/// buffer dtype is stamped BF16 so the selector/router GEMMs read it as BF16).
fn upload_bf16_rows(
    gpu: &mut Gpu,
    bytes: &[u8],
    m: usize,
    k: usize,
    name: &str,
) -> Result<WeightTensor, String> {
    let mut buf = gpu
        .upload_raw(bytes, &[m, k])
        .map_err(|e| format!("raon1_mini weights: upload {name}: {e:?}"))?;
    buf.dtype = DType::BF16;
    ROWS_BYTES_RESIDENT.fetch_add(buf.buf.size(), Ordering::Relaxed);
    ROWS_BYTES_AS_F32.fetch_add(m * k * 4, Ordering::Relaxed);
    Ok(rows_weight_tensor(buf, DType::BF16, m, k))
}

/// Router / mHC-selector weight `[m, k]` from F32 values (widened from F16/F32
/// storage): resident as BF16 (half the bytes, narrowing is exact) when every
/// value is exactly BF16-representable, which holds for the BF16 source
/// checkpoint. A tensor with any inexact value stays F32 and is logged; the
/// selector/router GEMMs run it on `gemm_f32_batched`, so values never change.
/// The raon_oracle uploads its synthetic cases through this function too.
pub fn upload_rows_weight(
    gpu: &mut Gpu,
    data: &[f32],
    m: usize,
    k: usize,
    name: &str,
) -> Result<WeightTensor, String> {
    if data.len() != m * k {
        return Err(format!(
            "raon1_mini weights: {name}: {} elements, expected {}",
            data.len(),
            m * k
        ));
    }
    match bf16_exact_image(data) {
        Ok(bits) => {
            let bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
            let w = upload_bf16_rows(gpu, &bytes, m, k, name)?;
            ROWS_BF16_NARROWED.fetch_add(1, Ordering::Relaxed);
            Ok(w)
        }
        Err((first, count)) => {
            eprintln!(
                "raon1_mini weights: {name}: {count} of {} values are not exactly BF16-representable                  (first at element {first}: {:#010x}); keeping this tensor F32",
                data.len(),
                data[first].to_bits()
            );
            ROWS_F32_KEPT.fetch_add(1, Ordering::Relaxed);
            let buf = upload_f32_named(gpu, data, &[m, k], name)?;
            ROWS_BYTES_RESIDENT.fetch_add(buf.buf.size(), Ordering::Relaxed);
            ROWS_BYTES_AS_F32.fetch_add(m * k * 4, Ordering::Relaxed);
            Ok(rows_weight_tensor(buf, DType::F32, m, k))
        }
    }
}

/// Router weight from its stored bytes: BF16 (qt16) uploads as-is, byte for
/// byte; F16 / F32 artifacts widen and narrow through [`upload_rows_weight`].
fn load_router_weight(
    gpu: &mut Gpu,
    qt: u8,
    data: &[u8],
    m: usize,
    k: usize,
    name: &str,
) -> Result<WeightTensor, String> {
    if qt == QT_BF16 {
        if data.len() != m * k * 2 {
            return Err(format!(
                "raon1_mini weights: {name}: {} BF16 bytes, expected {}",
                data.len(),
                m * k * 2
            ));
        }
        let w = upload_bf16_rows(gpu, data, m, k, name)?;
        ROWS_BF16_STORED.fetch_add(1, Ordering::Relaxed);
        return Ok(w);
    }
    let f = widen_f32(qt, data, m * k, name)?;
    upload_rows_weight(gpu, &f, m, k, name)
}

/// Shared-expert and dense-FFN projections: the always-active FFN weights
/// (every token reads them, unlike a routed expert). `model.layers.N.mlp.*`
/// (dense layer 0) and `model.layers.N.moe.shared_experts.*`.
fn is_always_active_ffn(name: &str) -> bool {
    name.ends_with("_proj.weight")
        && (name.contains(".mlp.") || name.contains(".moe.shared_experts."))
}

/// Counters of the BF16 always-active-FFN load (reported once per load).
static BF16_FFN_TENSORS: AtomicUsize = AtomicUsize::new(0);
static BF16_FFN_VALUES: AtomicUsize = AtomicUsize::new(0);
static BF16_FFN_INEXACT: AtomicUsize = AtomicUsize::new(0);

/// F16 little-endian image of a BF16 `[m, k]` weight (every BF16 value in the
/// F16 normal range is exactly representable: 8 mantissa bits into 10) and the
/// Q8F16 (`qt3`, 32-element blocks, f16 scale `amax/127`) decode copy of the
/// same values. Returns `(f16 bytes, q8 bytes, values not exactly
/// representable in F16)`. A BF16 value that overflows F16 is an error.
fn bf16_ffn_images(
    data: &[u8],
    m: usize,
    k: usize,
    name: &str,
) -> Result<(Vec<u8>, Vec<u8>, usize), String> {
    if k % 32 != 0 || data.len() != m * k * 2 {
        return Err(format!(
            "raon1_mini weights: {name}: BF16 FFN projection needs [m, K%32==0] ({m}x{k}, {} B)",
            data.len()
        ));
    }
    let mut f16 = Vec::with_capacity(data.len());
    let mut q8 = Vec::with_capacity(m * (k / 32) * 34);
    let mut inexact = 0usize;
    let mut block = [0f32; 32];
    for (bi, chunk) in data.chunks_exact(64).enumerate() {
        for (i, c) in chunk.chunks_exact(2).enumerate() {
            let v = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16);
            let h = f32_to_f16(v);
            let back = f16_to_f32(h);
            if !back.is_finite() || !v.is_finite() {
                return Err(format!(
                    "raon1_mini weights: {name}: BF16 value {v} at element {} does not fit F16",
                    bi * 32 + i
                ));
            }
            if back.to_bits() != v.to_bits() {
                inexact += 1;
            }
            f16.extend_from_slice(&h.to_le_bytes());
            block[i] = v;
        }
        let max_abs = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = max_abs / 127.0;
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        q8.extend_from_slice(&f32_to_f16(scale).to_le_bytes());
        for v in block {
            q8.push((v * inv).round().clamp(-128.0, 127.0) as i8 as u8);
        }
    }
    Ok((f16, q8, inexact))
}

/// One projection from the artifact. A BF16 always-active-FFN weight comes
/// back as its exact F16 image (prefill) plus a Q8_0 decode copy.
fn load_proj(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    m: usize,
    k: usize,
) -> Result<(WeightTensor, Option<WeightTensor>), String> {
    let (mut wt, decode) = read_bytes(hfq, name, |info, data| match info.quant_type {
        qt @ (QT_MQ4V2 | QT_Q8F16) => dequant_weight_raw(gpu, qt, data, m, k, MemoryTarget::Device)
            .map(|w| (w, None))
            .map_err(|e| format!("raon1_mini weights: upload {name}: {e:?}")),
        QT_BF16 if is_always_active_ffn(name) => {
            let (f16, q8, inexact) = bf16_ffn_images(data, m, k, name)?;
            let w = dequant_weight_raw(gpu, QT_F16, &f16, m, k, MemoryTarget::Device)
                .map_err(|e| format!("raon1_mini weights: upload {name} (F16): {e:?}"))?;
            let d = match dequant_weight_raw(gpu, QT_Q8F16, &q8, m, k, MemoryTarget::Device) {
                Ok(d) => d,
                Err(e) => {
                    w.free_all(gpu);
                    return Err(format!(
                        "raon1_mini weights: upload {name} (Q8_0 decode copy): {e:?}"
                    ));
                }
            };
            BF16_FFN_TENSORS.fetch_add(1, Ordering::Relaxed);
            BF16_FFN_VALUES.fetch_add(m * k, Ordering::Relaxed);
            BF16_FFN_INEXACT.fetch_add(inexact, Ordering::Relaxed);
            Ok((w, Some(d)))
        }
        qt @ (QT_F16 | QT_BF16 | QT_F32) => {
            let f = widen_f32(qt, data, m * k, name)?;
            f32_weight(gpu, &f, m, k, name).map(|w| (w, None))
        }
        other => Err(format!(
            "raon1_mini weights: {name}: unsupported projection quant type {other}"
        )),
    })?;
    if wt.gpu_dtype.supports_awq_sidecar() {
        match read_awq_host(hfq, name, k) {
            Ok(Some(v)) => match upload_f32_named(gpu, &v, &[k], &awq_sidecar_name(name)) {
                Ok(t) => wt.awq_scale = Some(t),
                Err(e) => {
                    wt.free_all(gpu);
                    if let Some(d) = decode {
                        d.free_all(gpu);
                    }
                    return Err(e);
                }
            },
            Ok(None) => {}
            Err(e) => {
                wt.free_all(gpu);
                if let Some(d) = decode {
                    d.free_all(gpu);
                }
                return Err(e);
            }
        }
    }
    Ok((wt, decode))
}

/// `gate`/`up` of one dense FFN must share an AWQ scale (one rotated
/// activation): both absent, or both present with identical values.
fn check_pair_awq_host(hfq: &HfqFile, gate: &str, up: &str, k: usize) -> Result<(), String> {
    let g = read_awq_host(hfq, gate, k)?;
    let u = read_awq_host(hfq, up, k)?;
    if g != u {
        return Err(format!(
            "raon1_mini weights: {gate}/{up}: gate/up AWQ sidecars differ (they consume one rotated activation)"
        ));
    }
    Ok(())
}

fn load_pool_item(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    cfg: &RaonConfig,
    spec: &TensorSpec,
) -> Result<Option<Loaded>, String> {
    let name = spec.name.as_str();
    Ok(Some(match spec.class {
        TensorClass::ExpertGate | TensorClass::ExpertUp | TensorClass::ExpertDown => {
            return Ok(None)
        }
        TensorClass::Embedding => {
            let (buf, fmt) = read_bytes(hfq, name, |info, data| {
                load_embedding(gpu, info.quant_type, data, cfg.vocab, cfg.hidden)
                    .map_err(|e| format!("raon1_mini weights: upload {name}: {e:?}"))
            })?;
            Loaded::Embed(buf, fmt)
        }
        TensorClass::Proj => match load_proj(hfq, gpu, name, spec.shape[0], spec.shape[1])? {
            (w, None) => Loaded::Mat(w),
            (w, Some(d)) => Loaded::MatDecode(w, d),
        },
        TensorClass::Router => Loaded::Mat(read_bytes(hfq, name, |info, data| {
            load_router_weight(
                gpu,
                info.quant_type,
                data,
                spec.shape[0],
                spec.shape[1],
                name,
            )
        })?),
        TensorClass::F32Vec => {
            let f = read_bytes(hfq, name, |info, data| {
                widen_f32(info.quant_type, data, spec.numel(), name)
            })?;
            Loaded::Vec(upload_f32_named(gpu, &f, &spec.shape, name)?)
        }
        TensorClass::HostF32 => {
            let n = spec.numel();
            Loaded::Host(read_bytes(hfq, name, |info, data| {
                widen_f32(info.quant_type, data, n, name)
            })?)
        }
    }))
}

fn load_mhc_site(
    gpu: &mut Gpu,
    pool: &mut Pool,
    p: &str,
    cfg: &RaonConfig,
) -> Result<MhcSiteWeights, String> {
    let n = cfg.mhc_streams;
    let w = n * cfg.hidden;
    let m = (n - 1) * (n - 1);
    let pre = pool.take_host(&format!("{p}.phi_pre"));
    let post = pool.take_host(&format!("{p}.phi_post"));
    let z = pool.take_host(&format!("{p}.res.phi_z"));
    let mut sel = Vec::with_capacity(MHC_CONTROLS * w);
    sel.extend_from_slice(&pre);
    sel.extend_from_slice(&post);
    sel.extend_from_slice(&z);
    let scalar = |pool: &mut Pool, k: &str| pool.take_host(&format!("{p}.{k}"))[0];
    let alpha_pre = scalar(pool, "alpha_pre");
    let alpha_post = scalar(pool, "alpha_post");
    let alpha_z = scalar(pool, "res.alpha_z");
    let eta = scalar(pool, "res.eta");
    let mut b_pre = [0f32; MHC_STREAMS];
    b_pre.copy_from_slice(&pool.take_host(&format!("{p}.b_pre")));
    let mut b_post = [0f32; MHC_STREAMS];
    b_post.copy_from_slice(&pool.take_host(&format!("{p}.b_post")));
    let mut b_z = [0f32; (MHC_STREAMS - 1) * (MHC_STREAMS - 1)];
    b_z.copy_from_slice(&pool.take_host(&format!("{p}.res.b_z")));
    debug_assert_eq!(b_z.len(), m);
    let mut params = vec![0f32; MHC_PARAM_LEN];
    params[MHC_PARAM_ALPHA_PRE] = alpha_pre;
    params[MHC_PARAM_ALPHA_POST] = alpha_post;
    params[MHC_PARAM_ALPHA_Z] = alpha_z;
    params[MHC_PARAM_ETA] = eta;
    params[MHC_PARAM_B_PRE..MHC_PARAM_B_POST].copy_from_slice(&b_pre);
    params[MHC_PARAM_B_POST..MHC_PARAM_B_Z].copy_from_slice(&b_post);
    params[MHC_PARAM_B_Z..].copy_from_slice(&b_z);
    let selector = upload_rows_weight(gpu, &sel, MHC_CONTROLS, w, &format!("{p}.selector"))?;
    let params_t = match upload_f32_named(gpu, &params, &[MHC_PARAM_LEN], &format!("{p}.params")) {
        Ok(t) => t,
        Err(e) => {
            selector.free_all(gpu);
            return Err(e);
        }
    };
    Ok(MhcSiteWeights {
        selector,
        params: params_t,
        alpha_pre,
        alpha_post,
        alpha_z,
        eta,
        b_pre,
        b_post,
        b_z,
    })
}

fn take_gated(pool: &mut Pool, prefix: &str) -> GatedFfnWeights {
    let (gate, gate_dec) = pool.take_ffn_mat(&format!("{prefix}.gate_proj.weight"));
    let (up, up_dec) = pool.take_ffn_mat(&format!("{prefix}.up_proj.weight"));
    let (down, down_dec) = pool.take_ffn_mat(&format!("{prefix}.down_proj.weight"));
    GatedFfnWeights {
        gate,
        up,
        down,
        decode: [gate_dec, up_dec, down_dec],
        iu4_symmetric: false,
    }
}

/// Whether every qt44 weight of a gated FFN (shared expert or dense MLP) has symmetric group
/// headers (`zp == -8 * sc`, the contract of the grouped symmetric quantized-activation cores),
/// checked on the device with the route's header checker. `false` off gfx1201, when the FFN has no
/// qt44 weight, or when the checker is unavailable; never an error (the quantized tiers simply
/// stay off for this FFN). Non-qt44 weights (Q8_0) are not covered by the verdict.
fn verify_gated_symmetric(gpu: &mut Gpu, g: &GatedFfnWeights) -> bool {
    let weights = [&g.gate, &g.up, &g.down];
    if gpu.arch != "gfx1201" || !weights.iter().any(|w| w.gpu_dtype == DType::MQ4G256V2) {
        return false;
    }
    for w in weights {
        if w.gpu_dtype != DType::MQ4G256V2 {
            continue;
        }
        let Ok(table) = ptr_table(gpu, &[w.buf.buf.as_ptr() as u64], "gated ffn sym check") else {
            return false;
        };
        let ok = matches!(gpu.qwen4_moe_sym_check(&table, w.m, w.k, 1, 136), Ok(true));
        free_t(gpu, table);
        if !ok {
            return false;
        }
    }
    true
}

fn ptr_table(gpu: &mut Gpu, ptrs: &[u64], label: &str) -> Result<GpuTensor, String> {
    let t = gpu
        .alloc_tensor(&[2 * ptrs.len()], DType::F32)
        .map_err(|e| format!("raon1_mini weights: alloc {label}: {e:?}"))?;
    let bytes: Vec<u8> = ptrs.iter().flat_map(|p| p.to_ne_bytes()).collect();
    if let Err(e) = gpu.hip.memcpy_htod(&t.buf, &bytes) {
        free_t(gpu, t);
        return Err(format!("raon1_mini weights: htod {label}: {e:?}"));
    }
    Ok(t)
}

struct ExpertAwqHost {
    gate_up: Option<Vec<f32>>,
    down: Vec<Option<Vec<f32>>>,
}

/// Read and cross-check the MoE layer's AWQ sidecars on the host, before any
/// expert is uploaded. Enforces the layer-level contracts of the module docs.
fn check_moe_awq(hfq: &HfqFile, cfg: &RaonConfig, m: &str) -> Result<ExpertAwqHost, String> {
    let c = cfg.hidden;
    let mut gate_up: Option<Vec<f32>> = None;
    let mut seen_gate_up = 0usize;
    let mut check_gu = |hfq: &HfqFile, name: String| -> Result<(), String> {
        match read_awq_host(hfq, &name, c)? {
            None => Ok(()),
            Some(v) => {
                seen_gate_up += 1;
                match &gate_up {
                    None => {
                        gate_up = Some(v);
                        Ok(())
                    }
                    Some(first) if *first == v => Ok(()),
                    Some(_) => Err(format!(
                        "raon1_mini weights: {m}: AWQ sidecar of {name} differs from the layer's \
                         gate/up scale (all routed gate/up and the shared gate/up consume one rotated activation)"
                    )),
                }
            }
        }
    };
    for e in 0..cfg.n_experts {
        check_gu(hfq, format!("{m}.experts.{e}.gate_proj.weight"))?;
        check_gu(hfq, format!("{m}.experts.{e}.up_proj.weight"))?;
    }
    check_gu(hfq, format!("{m}.shared_experts.gate_proj.weight"))?;
    check_gu(hfq, format!("{m}.shared_experts.up_proj.weight"))?;
    // A Q8F16/BF16 shared gate/up consumes the plain activation: no sidecar.
    let shared_gu = |name: &str| {
        usize::from(
            hfq.find_tensor_info(name)
                .is_some_and(|i| i.quant_type == QT_MQ4V2),
        )
    };
    let expected_gu = 2 * cfg.n_experts
        + shared_gu(&format!("{m}.shared_experts.gate_proj.weight"))
        + shared_gu(&format!("{m}.shared_experts.up_proj.weight"));
    if seen_gate_up != 0 && seen_gate_up != expected_gu {
        return Err(format!(
            "raon1_mini weights: {m}: partial gate/up AWQ coverage ({seen_gate_up}/{expected_gu}); \
             sidecars are all-or-none per layer"
        ));
    }
    let mut down = Vec::with_capacity(cfg.n_experts);
    let mut have = 0usize;
    for e in 0..cfg.n_experts {
        let name = format!("{m}.experts.{e}.down_proj.weight");
        let v = read_awq_host(hfq, &name, cfg.moe_down_k)?;
        if let Some(ref v) = v {
            have += 1;
            if v[cfg.moe_inter..].iter().any(|x| *x != 1.0) {
                return Err(format!(
                    "raon1_mini weights: {}: padded channels [{}, {}) must be exactly 1.0",
                    awq_sidecar_name(&name),
                    cfg.moe_inter,
                    cfg.moe_down_k
                ));
            }
        }
        down.push(v);
    }
    if have != 0 && have != cfg.n_experts {
        return Err(format!(
            "raon1_mini weights: {m}: partial expert down AWQ coverage ({have}/{}); all-or-none per layer",
            cfg.n_experts
        ));
    }
    if let Some(v) = read_awq_host(
        hfq,
        &format!("{m}.shared_experts.down_proj.weight"),
        cfg.shared_down_k(),
    )? {
        if v[cfg.shared_inter()..].iter().any(|x| *x != 1.0) {
            return Err(format!(
                "raon1_mini weights: {m}.shared_experts.down_proj: padded AWQ channels must be exactly 1.0"
            ));
        }
    }
    Ok(ExpertAwqHost { gate_up, down })
}

fn load_experts(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    cfg: &RaonConfig,
    layer: usize,
) -> Result<Vec<ExpertWeights>, String> {
    let m = format!("model.layers.{layer}.moe");
    let mut experts: Vec<ExpertWeights> = Vec::with_capacity(cfg.n_experts);
    for e in 0..cfg.n_experts {
        let ep = format!("{m}.experts.{e}");
        let gname = format!("{ep}.gate_proj.weight");
        let uname = format!("{ep}.up_proj.weight");
        let dname = format!("{ep}.down_proj.weight");
        let res = (|| -> Result<ExpertWeights, String> {
            let fused = read_bytes(hfq, &gname, |_, g| {
                read_bytes(hfq, &uname, |_, u| {
                    let mut f = Vec::with_capacity(g.len() + u.len());
                    f.extend_from_slice(g);
                    f.extend_from_slice(u);
                    Ok(f)
                })
            })?;
            let gate_up_buf = gpu
                .upload_raw(&fused, &[fused.len()])
                .map_err(|er| format!("raon1_mini weights: upload {ep} gate_up: {er:?}"))?;
            let gate_up = WeightTensor {
                buf: gate_up_buf,
                gpu_dtype: DType::MQ4G256V2,
                m: 2 * cfg.moe_inter,
                k: cfg.hidden,
                row_stride: 0,
                paro: None,
                awq_scale: None,
                lloyd_lut_e4m3: None,
                lloyd_lut_f16: None,
                lloyd_lut_c16: None,
            };
            let down_buf = match read_bytes(hfq, &dname, |_, d| {
                gpu.upload_raw(d, &[d.len()])
                    .map_err(|er| format!("raon1_mini weights: upload {dname}: {er:?}"))
            }) {
                Ok(b) => b,
                Err(er) => {
                    gate_up.free_all(gpu);
                    return Err(er);
                }
            };
            Ok(ExpertWeights {
                gate_up,
                down: WeightTensor {
                    buf: down_buf,
                    gpu_dtype: DType::MQ4G256V2,
                    m: cfg.hidden,
                    k: cfg.moe_down_k,
                    row_stride: 0,
                    paro: None,
                    awq_scale: None,
                    lloyd_lut_e4m3: None,
                    lloyd_lut_f16: None,
                    lloyd_lut_c16: None,
                },
            })
        })();
        match res {
            Ok(x) => experts.push(x),
            Err(er) => {
                for x in experts {
                    x.free_gpu(gpu);
                }
                return Err(er);
            }
        }
    }
    Ok(experts)
}

/// Build the sealed identity + pointer tables over resident experts.
fn seal_experts(
    gpu: &mut Gpu,
    layer: usize,
    experts: &[ExpertWeights],
    down_awq_ptrs: Option<&GpuTensor>,
) -> Result<(GpuTensor, GpuTensor, ExpertTable, ExpertBindingCache), String> {
    let mut metadata = Vec::with_capacity(experts.len());
    for (e, expert) in experts.iter().enumerate() {
        let resource = |name: &str, w: &WeightTensor| {
            let row = w.gpu_dtype.row_bytes(w.k).ok_or_else(|| {
                format!("raon1_mini L{layer}: no row layout for {:?}", w.gpu_dtype)
            })?;
            ExpertResource::new(
                format!("model.layers.{layer}.moe.experts.{e}.{name}"),
                "raon1-resident-v1",
                vec![w.m, w.k],
                w.gpu_dtype,
                w.m * row,
                row,
                16,
                dtype_rotation_plan(w.gpu_dtype),
            )
            .map_err(|err| format!("raon1_mini L{layer}: expert {e} {name}: {err:?}"))
        };
        let resources = ExpertResources::fused(
            resource("gate_up", &expert.gate_up)?,
            resource("down", &expert.down)?,
        )
        .map_err(|err| format!("raon1_mini L{layer}: expert {e}: {err:?}"))?;
        metadata.push(
            ExpertMetadata::new(e, 0, e, resources)
                .map_err(|err| format!("raon1_mini L{layer}: expert {e}: {err:?}"))?,
        );
    }
    let table = ExpertTable::new(metadata)
        .map_err(|err| format!("raon1_mini L{layer}: expert table: {err:?}"))?;
    let mut cache = table
        .prepare_binding(0, 1, gpu.device_id)
        .map_err(|err| format!("raon1_mini L{layer}: expert binding: {err:?}"))?;
    let gu_ptrs: Vec<u64> = experts
        .iter()
        .map(|x| x.gate_up.buf.buf.as_ptr() as u64)
        .collect();
    let dn_ptrs: Vec<u64> = experts
        .iter()
        .map(|x| x.down.buf.buf.as_ptr() as u64)
        .collect();
    let gu = ptr_table(gpu, &gu_ptrs, "gate_up ptrs")?;
    let dn = match ptr_table(gpu, &dn_ptrs, "down ptrs") {
        Ok(t) => t,
        Err(e) => {
            free_t(gpu, gu);
            return Err(e);
        }
    };
    if let Err(err) = cache.bind_live(
        &table,
        &ResidentExperts(experts),
        &gu,
        &dn,
        down_awq_ptrs,
        None,
    ) {
        free_t(gpu, gu);
        free_t(gpu, dn);
        return Err(format!("raon1_mini L{layer}: bind live experts: {err:?}"));
    }
    Ok((gu, dn, table, cache))
}

/// Whether every routed gate/up and down group header obeys `zp == -8 * sc` (the symmetric qt44
/// contract the grouped W4A8 / W4A4 cores read: `w = sc * (u - 8)`), checked on the device with the
/// route's header checker. `false` on archs that admit no quantized grouped tier
/// ([`grouped_sym_tier_admits`]), for non-qt44 experts, or when the checker is unavailable; never an
/// error (the route simply stays off).
fn verify_symmetric_headers(
    gpu: &mut Gpu,
    experts: &[ExpertWeights],
    gu: &GpuTensor,
    dn: &GpuTensor,
) -> bool {
    let Some(first) = experts.first() else {
        return false;
    };
    let quantized_tier = grouped_sym_tier_admits(&gpu.arch, PrefillActTier::A8)
        || grouped_sym_tier_admits(&gpu.arch, PrefillActTier::A4);
    if !quantized_tier
        || !experts.iter().all(|x| {
            x.gate_up.gpu_dtype == DType::MQ4G256V2 && x.down.gpu_dtype == DType::MQ4G256V2
        })
    {
        return false;
    }
    let n = experts.len();
    let gu_ok = gpu.qwen4_moe_sym_check(gu, first.gate_up.m, first.gate_up.k, n, 136);
    let dn_ok = gpu.qwen4_moe_sym_check(dn, first.down.m, first.down.k, n, 136);
    matches!((gu_ok, dn_ok), (Ok(true), Ok(true)))
}

fn build_moe(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    cfg: &RaonConfig,
    pool: &mut Pool,
    layer: usize,
) -> Result<MoeFfnWeights, String> {
    let m = format!("model.layers.{layer}.moe");
    let awq = check_moe_awq(hfq, cfg, &m)?;
    let experts = load_experts(hfq, gpu, cfg, layer)?;
    let mut owners: Vec<GpuTensor> = Vec::new();
    let cleanup = |gpu: &mut Gpu, experts: Vec<ExpertWeights>, owners: Vec<GpuTensor>| {
        for x in experts {
            x.free_gpu(gpu);
        }
        for t in owners {
            free_t(gpu, t);
        }
    };
    // Per-expert down AWQ tensors + pointer table.
    let mut down_ptrs_table: Option<GpuTensor> = None;
    if awq.down.iter().all(|d| d.is_some()) && cfg.n_experts > 0 {
        for (e, d) in awq.down.iter().enumerate() {
            let v = d.as_ref().expect("checked");
            match upload_f32_named(
                gpu,
                v,
                &[cfg.moe_down_k],
                &format!("{m}.experts.{e}.down awq"),
            ) {
                Ok(t) => owners.push(t),
                Err(er) => {
                    cleanup(gpu, experts, owners);
                    return Err(er);
                }
            }
        }
        let ptrs: Vec<u64> = owners.iter().map(|t| t.buf.as_ptr() as u64).collect();
        match ptr_table(gpu, &ptrs, "down awq ptrs") {
            Ok(t) => down_ptrs_table = Some(t),
            Err(er) => {
                cleanup(gpu, experts, owners);
                return Err(er);
            }
        }
    }
    let gate_up_awq = match &awq.gate_up {
        Some(v) => match upload_f32_named(gpu, v, &[cfg.hidden], &format!("{m} gate_up awq")) {
            Ok(t) => Some(t),
            Err(er) => {
                if let Some(t) = down_ptrs_table {
                    free_t(gpu, t);
                }
                cleanup(gpu, experts, owners);
                return Err(er);
            }
        },
        None => None,
    };
    let sealed = seal_experts(gpu, layer, &experts, down_ptrs_table.as_ref());
    let (gu, dn, table, cache) = match sealed {
        Ok(x) => x,
        Err(er) => {
            if let Some(t) = gate_up_awq {
                free_t(gpu, t);
            }
            if let Some(t) = down_ptrs_table {
                free_t(gpu, t);
            }
            cleanup(gpu, experts, owners);
            return Err(er);
        }
    };
    let mut shared = take_gated(pool, &format!("{m}.shared_experts"));
    // The qt44 shared expert's gate/up scale must be the layer's (values are
    // verified identical by `check_moe_awq`): both present or both absent.
    // Q8F16/BF16 shared projections consume the plain activation and carry none.
    let shared_ok = if (shared.gate.gpu_dtype == DType::MQ4G256V2
        && shared.gate.awq_scale.is_some() != gate_up_awq.is_some())
        || (shared.up.gpu_dtype == DType::MQ4G256V2
            && shared.up.awq_scale.is_some() != gate_up_awq.is_some())
    {
        Err(format!(
            "raon1_mini weights: {m}: shared gate/up AWQ presence differs from routed experts"
        ))
    } else {
        Ok(())
    };
    if let Err(er) = shared_ok {
        shared.free_gpu(gpu);
        free_t(gpu, gu);
        free_t(gpu, dn);
        if let Some(t) = gate_up_awq {
            free_t(gpu, t);
        }
        if let Some(t) = down_ptrs_table {
            free_t(gpu, t);
        }
        cleanup(gpu, experts, owners);
        return Err(er);
    }
    let iu4_symmetric = verify_symmetric_headers(gpu, &experts, &gu, &dn);
    shared.iu4_symmetric = verify_gated_symmetric(gpu, &shared);
    Ok(MoeFfnWeights {
        router: pool.take_mat(&format!("{m}.gate.weight")),
        e_bias: pool.take_vec(&format!("{m}.gate.e_score_correction_bias")),
        shared,
        experts,
        expert_gate_up_ptrs: gu,
        expert_down_ptrs: dn,
        gate_up_awq,
        expert_down_awq_ptrs: down_ptrs_table,
        expert_down_awq: owners,
        iu4_symmetric,
        table,
        cache,
    })
}

/// Every pool entry of one layer present with the declared kind.
fn layer_ready(pool: &Pool, specs_by_layer: &[&TensorSpec]) -> bool {
    specs_by_layer
        .iter()
        .filter(|s| {
            !matches!(
                s.class,
                TensorClass::ExpertGate | TensorClass::ExpertUp | TensorClass::ExpertDown
            )
        })
        .all(|s| pool.kind_ok(s))
}

fn build_layer(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    cfg: &RaonConfig,
    pool: &mut Pool,
    layer: usize,
) -> Result<RaonLayerWeights, String> {
    let lp = format!("model.layers.{layer}");
    let a = format!("{lp}.self_attn");
    // The MoE (fallible, large) goes first: nothing else of this layer has
    // been taken from the pool yet, so a failure leaks nothing.
    let ffn = match cfg.ffn_kinds[layer] {
        FfnKind::Moe => RaonFfn::Moe(build_moe(hfq, gpu, cfg, pool, layer)?),
        FfnKind::Dense => {
            let mlp = format!("{lp}.mlp");
            check_pair_awq_host(
                hfq,
                &format!("{mlp}.gate_proj.weight"),
                &format!("{mlp}.up_proj.weight"),
                cfg.hidden,
            )?;
            let mut g = take_gated(pool, &mlp);
            g.iu4_symmetric = verify_gated_symmetric(gpu, &g);
            RaonFfn::Dense(g)
        }
    };
    let attn_mhc = match load_mhc_site(gpu, pool, &format!("{lp}.self_attn_raon_mhc"), cfg) {
        Ok(s) => s,
        Err(e) => {
            ffn.free_gpu(gpu);
            return Err(e);
        }
    };
    let ffn_mhc = match load_mhc_site(gpu, pool, &format!("{lp}.mlp_raon_mhc"), cfg) {
        Ok(s) => s,
        Err(e) => {
            attn_mhc.free_gpu(gpu);
            ffn.free_gpu(gpu);
            return Err(e);
        }
    };
    let mixer = match cfg.layer_kinds[layer] {
        LayerKind::Rda => RaonMixer::Rda(RdaWeights {
            q_proj: pool.take_mat(&format!("{a}.q_proj.weight")),
            k_proj: pool.take_mat(&format!("{a}.k_proj.weight")),
            v_proj: pool.take_mat(&format!("{a}.v_proj.weight")),
            g_proj: pool.take_mat(&format!("{a}.g_proj.weight")),
            o_proj: pool.take_mat(&format!("{a}.o_proj.weight")),
            f_a_proj: pool.take_mat(&format!("{a}.f_a_proj.weight")),
            f_b_proj: pool.take_mat(&format!("{a}.f_b_proj.weight")),
            b_proj: pool.take_mat(&format!("{a}.b_proj.weight")),
            rda_w_proj: pool.take_mat(&format!("{a}.rda_w_proj.weight")),
            q_conv: pool.take_vec(&format!("{a}.q_conv1d.weight")),
            k_conv: pool.take_vec(&format!("{a}.k_conv1d.weight")),
            v_conv: pool.take_vec(&format!("{a}.v_conv1d.weight")),
            a_log: pool.take_vec(&format!("{a}.A_log")),
            dt_bias: pool.take_vec(&format!("{a}.dt_bias")),
            o_norm: pool.take_vec(&format!("{a}.o_norm.weight")),
        }),
        LayerKind::Mla => RaonMixer::Mla(MlaWeights {
            q_a_proj: pool.take_mat(&format!("{a}.q_a_proj.weight")),
            q_a_norm: pool.take_vec(&format!("{a}.q_a_layernorm.weight")),
            q_b_proj: pool.take_mat(&format!("{a}.q_b_proj.weight")),
            kv_a_proj: pool.take_mat(&format!("{a}.kv_a_proj_with_mqa.weight")),
            kv_a_norm: pool.take_vec(&format!("{a}.kv_a_layernorm.weight")),
            kv_b_proj: pool.take_mat(&format!("{a}.kv_b_proj.weight")),
            g_proj: pool.take_mat(&format!("{a}.g_proj.weight")),
            o_proj: pool.take_mat(&format!("{a}.o_proj.weight")),
        }),
    };
    Ok(RaonLayerWeights {
        input_norm: pool.take_vec(&format!("{lp}.input_layernorm.weight")),
        post_attn_norm: pool.take_vec(&format!("{lp}.post_attention_layernorm.weight")),
        attn_mhc,
        ffn_mhc,
        mixer,
        ffn,
    })
}

/// Load every tensor of a `raon1_mini` HFQ artifact.
///
/// Refuses (before the first allocation) a missing tensor, a wrong shape,
/// an inadmissible quant type or payload size, an unpadded `[2560, 1088]`
/// down projection, and any tensor the inventory does not declare. Requires
/// the symmetric-MQ4V2 marker (`"mq4v2.symmetric": 1`, header-sampled at
/// `HfqFile::open`) whenever qt44 tensors are present. On any later error every
/// buffer uploaded so far is freed.
pub fn load_raon_weights(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    cfg: &RaonConfig,
) -> Result<RaonWeights, String> {
    cfg.validate()?;
    let specs = tensor_inventory(cfg);
    let index: Vec<IndexEntry> = hfq.tensors().iter().map(IndexEntry::from_info).collect();
    check_inventory(&specs, &index)?;
    BF16_FFN_TENSORS.store(0, Ordering::Relaxed);
    BF16_FFN_VALUES.store(0, Ordering::Relaxed);
    BF16_FFN_INEXACT.store(0, Ordering::Relaxed);
    ROWS_BF16_STORED.store(0, Ordering::Relaxed);
    ROWS_BF16_NARROWED.store(0, Ordering::Relaxed);
    ROWS_F32_KEPT.store(0, Ordering::Relaxed);
    ROWS_BYTES_RESIDENT.store(0, Ordering::Relaxed);
    ROWS_BYTES_AS_F32.store(0, Ordering::Relaxed);
    if index.iter().any(|e| e.quant_type == QT_MQ4V2) && !hfq.mq4v2_symmetric() {
        return Err(
            "raon1_mini weights: qt44 tensors present but the artifact lacks the symmetric \
             marker (metadata `mq4v2.symmetric`=1); Raon requires symmetric MQ4V2 \
             (`hipfire quantize --mq4v2-symmetric`)"
                .into(),
        );
    }
    // The verified zero contract admits the device's symmetric-only routes
    // (A8/IU4 prefill); whether they run is the bundle's precision policy.
    gpu.mq4v2_symmetric = hfq.mq4v2_symmetric();

    // Stage 1: upload everything except routed experts into the pool.
    let mut pool = Pool::default();
    for spec in &specs {
        match load_pool_item(hfq, gpu, cfg, spec) {
            Ok(Some(item)) => {
                pool.items.insert(spec.name.clone(), item);
            }
            Ok(None) => {}
            Err(e) => {
                pool.free(gpu);
                return Err(e);
            }
        }
    }
    // Stage 2: layer assembly (inventory-consistency gate first).
    let mut by_layer: Vec<Vec<&TensorSpec>> = vec![Vec::new(); cfg.n_layers];
    for s in &specs {
        if let Some(rest) = s.name.strip_prefix("model.layers.") {
            if let Some(l) = rest.split('.').next().and_then(|n| n.parse::<usize>().ok()) {
                by_layer[l].push(s);
            }
        }
    }
    for (l, group) in by_layer.iter().enumerate() {
        if !layer_ready(&pool, group) {
            pool.free(gpu);
            return Err(format!(
                "raon1_mini weights: internal inventory/loader mismatch at layer {l}"
            ));
        }
    }
    let mut layers: Vec<RaonLayerWeights> = Vec::with_capacity(cfg.n_layers);
    for l in 0..cfg.n_layers {
        match build_layer(hfq, gpu, cfg, &mut pool, l) {
            Ok(layer) => layers.push(layer),
            Err(e) => {
                for built in layers {
                    built.free_gpu(gpu);
                }
                pool.free(gpu);
                return Err(e);
            }
        }
    }
    // Globals: verified present by the gate above (non-layer specs).
    for s in specs
        .iter()
        .filter(|s| !s.name.starts_with("model.layers."))
    {
        if !pool.kind_ok(s) {
            for built in layers {
                built.free_gpu(gpu);
            }
            pool.free(gpu);
            return Err(format!(
                "raon1_mini weights: internal inventory/loader mismatch at {}",
                s.name
            ));
        }
    }
    let (embed, embed_format) = match pool.items.remove("model.embed_tokens.weight") {
        Some(Loaded::Embed(t, f)) => (t, f),
        _ => unreachable!("embedding verified"),
    };
    let final_norm = pool.take_vec("model.norm.weight");
    let lm_head = pool.take_mat("lm_head.weight");
    debug_assert!(
        pool.items.is_empty() || pool.items.keys().all(|k| !pool.has("") || k.is_empty())
    );
    let leftover: Vec<String> = pool.items.keys().cloned().collect();
    if !leftover.is_empty() {
        // Every declared non-expert tensor must have been consumed by assembly.
        let w = RaonWeights {
            embed,
            embed_format,
            final_norm,
            lm_head,
            layers,
        };
        w.free_gpu(gpu);
        pool.free(gpu);
        return Err(format!(
            "raon1_mini weights: loaded-but-unassembled tensors: {}",
            list_cap(&leftover)
        ));
    }
    let bf16_tensors = BF16_FFN_TENSORS.swap(0, Ordering::Relaxed);
    if bf16_tensors != 0 {
        eprintln!(
            "raon1_mini weights: always-active FFN stored BF16: {bf16_tensors} tensors resident as \
             F16 images for prefill ({} of {} values not exactly F16-representable) + Q8_0 decode copies",
            BF16_FFN_INEXACT.swap(0, Ordering::Relaxed),
            BF16_FFN_VALUES.swap(0, Ordering::Relaxed)
        );
    }
    let (stored, narrowed, kept) = (
        ROWS_BF16_STORED.swap(0, Ordering::Relaxed),
        ROWS_BF16_NARROWED.swap(0, Ordering::Relaxed),
        ROWS_F32_KEPT.swap(0, Ordering::Relaxed),
    );
    let (resident, as_f32) = (
        ROWS_BYTES_RESIDENT.swap(0, Ordering::Relaxed),
        ROWS_BYTES_AS_F32.swap(0, Ordering::Relaxed),
    );
    eprintln!(
        "raon1_mini weights: router + mHC selector: {stored} read as stored BF16, {narrowed} narrowed \
         from F16/F32 (every value verified BF16-representable), {kept} kept F32; {resident} device \
         bytes resident ({as_f32} as F32, {} saved)",
        as_f32.saturating_sub(resident)
    );
    Ok(RaonWeights {
        embed,
        embed_format,
        final_norm,
        lm_head,
        layers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RaonConfig;

    fn released() -> RaonConfig {
        RaonConfig::from_json(&serde_json::json!({"model_type": "raon1_mini"})).unwrap()
    }

    /// Index entries a well-formed control artifact would carry for `specs`.
    fn good_index(specs: &[TensorSpec]) -> Vec<IndexEntry> {
        specs
            .iter()
            .map(|s| {
                let qt = match s.class {
                    TensorClass::Embedding => QT_Q8F16,
                    TensorClass::Proj => {
                        let k = *s.shape.last().unwrap();
                        if k % 256 == 0 {
                            QT_MQ4V2
                        } else {
                            QT_Q8F16
                        }
                    }
                    TensorClass::ExpertGate | TensorClass::ExpertUp | TensorClass::ExpertDown => {
                        QT_MQ4V2
                    }
                    _ => QT_BF16,
                };
                let (m, k) = match s.shape.as_slice() {
                    [] => (1, 1),
                    [n] => (1, *n),
                    [m, k] => (*m, *k),
                    _ => unreachable!(),
                };
                IndexEntry {
                    name: s.name.clone(),
                    quant_type: qt,
                    shape: s.shape.clone(),
                    data_size: payload_bytes(qt, m, k).unwrap(),
                }
            })
            .collect()
    }

    #[test]
    fn inventory_matches_released_checkpoint_counts() {
        let cfg = released();
        let specs = tensor_inventory(&cfg);
        // 19434 official tensors − 9216 weight_scale_inv (3072*3) = 10218 declared… derive exactly:
        // globals 3 + per layer (norms 2 + mhc 20) + RDA 15*18 + MLA 8*7 + dense 3 + moe (2+3) *24 + 128*3*24.
        let want = 3 + 25 * 22 + 18 * 15 + 7 * 8 + 3 + 24 * 5 + 24 * 128 * 3;
        assert_eq!(specs.len(), want);
        assert_eq!(specs.len(), 19434 - 24 * 128 * 3);
        let names: HashSet<&str> = specs.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names.len(), specs.len(), "duplicate declared names");
        // Layer 0: RDA + dense, no moe.
        assert!(names.contains("model.layers.0.self_attn.q_proj.weight"));
        assert!(names.contains("model.layers.0.mlp.down_proj.weight"));
        assert!(!names.contains("model.layers.0.moe.gate.weight"));
        // Layer 3 (zero-based) is MLA; layer 24 is the last MLA + MoE; no layer 25.
        assert!(names.contains("model.layers.3.self_attn.kv_b_proj.weight"));
        assert!(!names.contains("model.layers.3.self_attn.q_proj.weight"));
        assert!(names.contains("model.layers.24.self_attn.q_b_proj.weight"));
        assert!(!names.iter().any(|n| n.starts_with("model.layers.25.")));
        // Padded down K.
        let down = specs
            .iter()
            .find(|s| s.name == "model.layers.1.moe.experts.0.down_proj.weight")
            .unwrap();
        assert_eq!(down.shape, vec![2560, 1280]);
        let shared = specs
            .iter()
            .find(|s| s.name == "model.layers.1.moe.shared_experts.down_proj.weight")
            .unwrap();
        assert_eq!(shared.shape, vec![2560, 1280]);
        let qb = specs
            .iter()
            .find(|s| s.name == "model.layers.3.self_attn.q_b_proj.weight")
            .unwrap();
        assert_eq!(qb.shape, vec![7680, 576]);
    }

    #[test]
    fn inventory_covers_the_checkpoint_index_names() {
        // Cross-check against the official safetensors index when it is present on this host.
        let path = "/home/kaden/qcal/parents/raon-1-mini-preview/model.safetensors.index.json";
        let Ok(text) = std::fs::read_to_string(path) else {
            return;
        };
        let idx: serde_json::Value = serde_json::from_str(&text).unwrap();
        let map = idx["weight_map"].as_object().unwrap();
        let specs = tensor_inventory(&released());
        let declared: HashSet<&str> = specs.iter().map(|s| s.name.as_str()).collect();
        for name in &declared {
            assert!(
                map.contains_key(*name),
                "declared {name} not in official index"
            );
        }
        for name in map.keys() {
            if name.ends_with(".weight_scale_inv") {
                continue; // consumed by the quantizer, not part of the HFQ
            }
            assert!(
                declared.contains(name.as_str()),
                "official tensor {name} not declared"
            );
        }
    }

    #[test]
    fn well_formed_artifact_passes() {
        let specs = tensor_inventory(&released());
        check_inventory(&specs, &good_index(&specs)).unwrap();
    }

    #[test]
    fn missing_tensors_are_listed() {
        let specs = tensor_inventory(&released());
        let mut idx = good_index(&specs);
        idx.retain(|e| {
            e.name != "model.layers.7.self_attn.kv_b_proj.weight" && e.name != "lm_head.weight"
        });
        let err = check_inventory(&specs, &idx).unwrap_err();
        assert!(err.contains("missing 2 tensor(s)"), "{err}");
        assert!(
            err.contains("model.layers.7.self_attn.kv_b_proj.weight"),
            "{err}"
        );
        assert!(err.contains("lm_head.weight"), "{err}");
    }

    #[test]
    fn wrong_family_and_shapes_are_refused() {
        let cfg = released();
        let specs = tensor_inventory(&cfg);
        // Wrong mixer family: layer 3 is MLA, so an RDA q_proj there is unconsumed and kv_b is missing.
        let mut idx = good_index(&specs);
        idx.retain(|e| e.name != "model.layers.3.self_attn.kv_b_proj.weight");
        idx.push(IndexEntry {
            name: "model.layers.3.self_attn.q_proj.weight".into(),
            quant_type: QT_MQ4V2,
            shape: vec![5120, 2560],
            data_size: 5120 * 10 * 136,
        });
        let err = check_inventory(&specs, &idx).unwrap_err();
        assert!(
            err.contains("missing") && err.contains("unconsumed"),
            "{err}"
        );
        assert!(err.contains("layers.3.self_attn.q_proj.weight"), "{err}");
        // Unpadded routed down.
        let mut idx = good_index(&specs);
        let e = idx
            .iter_mut()
            .find(|e| e.name == "model.layers.2.moe.experts.5.down_proj.weight")
            .unwrap();
        e.shape = vec![2560, 1088];
        let err = check_inventory(&specs, &idx).unwrap_err();
        assert!(err.contains("unpadded K"), "{err}");
        // Wrong payload size.
        let mut idx = good_index(&specs);
        idx.iter_mut()
            .find(|e| {
                e.name == "model.layers.2.mlp.gate_proj.weight"
                    || e.name == "model.layers.0.mlp.gate_proj.weight"
            })
            .unwrap()
            .data_size -= 136;
        assert!(check_inventory(&specs, &idx)
            .unwrap_err()
            .contains("payload"));
        // qt44 on q_b (K=576 not a group multiple) cannot encode.
        let mut idx = good_index(&specs);
        let e = idx
            .iter_mut()
            .find(|e| e.name == "model.layers.3.self_attn.q_b_proj.weight")
            .unwrap();
        e.quant_type = QT_MQ4V2;
        assert!(check_inventory(&specs, &idx)
            .unwrap_err()
            .contains("cannot encode"));
        // Non-qt44 routed expert.
        let mut idx = good_index(&specs);
        let e = idx
            .iter_mut()
            .find(|e| e.name == "model.layers.1.moe.experts.0.gate_proj.weight")
            .unwrap();
        e.quant_type = QT_Q8F16;
        assert!(check_inventory(&specs, &idx)
            .unwrap_err()
            .contains("not admitted"));
    }

    #[test]
    fn raw_fp8_scale_tensors_are_unconsumed() {
        let specs = tensor_inventory(&released());
        let mut idx = good_index(&specs);
        idx.push(IndexEntry {
            name: "model.layers.1.moe.experts.0.gate_proj.weight_scale_inv".into(),
            quant_type: 0,
            shape: vec![1088, 80],
            data_size: 1088 * 80,
        });
        let err = check_inventory(&specs, &idx).unwrap_err();
        assert!(
            err.contains("weight_scale_inv") && err.contains("hipfire quantize"),
            "{err}"
        );
    }

    #[test]
    fn awq_sidecars_are_validated_and_not_unconsumed() {
        let specs = tensor_inventory(&released());
        let mut idx = good_index(&specs);
        let name = "model.layers.1.moe.experts.0.down_proj.awq_scale.weight";
        idx.push(IndexEntry {
            name: name.into(),
            quant_type: QT_F16,
            shape: vec![1280],
            data_size: 2560,
        });
        check_inventory(&specs, &idx).unwrap();
        // Wrong width.
        idx.last_mut().unwrap().shape = vec![1088];
        idx.last_mut().unwrap().data_size = 2176;
        assert!(check_inventory(&specs, &idx)
            .unwrap_err()
            .contains("sidecar must be F16"));
        // Sidecar on a Q8F16 weight is refused.
        let mut idx = good_index(&specs);
        idx.push(IndexEntry {
            name: "model.layers.3.self_attn.q_b_proj.awq_scale.weight".into(),
            quant_type: QT_F16,
            shape: vec![576],
            data_size: 1152,
        });
        assert!(check_inventory(&specs, &idx)
            .unwrap_err()
            .contains("only qt44"));
        // Orphan sidecar (no weight) is unconsumed.
        let mut idx = good_index(&specs);
        idx.push(IndexEntry {
            name: "model.layers.9.self_attn.nothing.awq_scale.weight".into(),
            quant_type: QT_F16,
            shape: vec![2560],
            data_size: 5120,
        });
        assert!(check_inventory(&specs, &idx)
            .unwrap_err()
            .contains("unconsumed"));
    }

    #[test]
    fn scalar_shapes_accept_rank0_or_one_element() {
        let specs = tensor_inventory(&released());
        let mut idx = good_index(&specs);
        let i = idx
            .iter()
            .position(|e| e.name == "model.layers.4.mlp_raon_mhc.res.eta")
            .unwrap();
        idx[i].shape = vec![1];
        check_inventory(&specs, &idx).unwrap();
        idx[i].shape = vec![2];
        assert!(check_inventory(&specs, &idx).is_err());
    }

    #[test]
    fn mhc_param_layout_is_dense() {
        assert_eq!(MHC_CONTROLS, 17);
        assert_eq!(MHC_PARAM_B_PRE, 4);
        assert_eq!(MHC_PARAM_B_POST, 8);
        assert_eq!(MHC_PARAM_B_Z, 12);
        assert_eq!(MHC_PARAM_LEN, 21);
    }

    /// Re-type the always-active FFN weights of `idx` (and drop their sidecars).
    fn retype_ffn(idx: &mut Vec<IndexEntry>, qt: u8) {
        for e in idx.iter_mut() {
            if is_always_active_ffn(&e.name) {
                let (m, k) = (e.shape[0], e.shape[1]);
                e.quant_type = qt;
                e.data_size = payload_bytes(qt, m, k).unwrap();
            }
        }
        idx.retain(|e| {
            !(e.name.ends_with(".awq_scale.weight") && {
                let w = e.name.replace(".awq_scale.weight", ".weight");
                is_always_active_ffn(&w)
            })
        });
    }

    #[test]
    fn always_active_ffn_classifier_covers_shared_and_dense_only() {
        assert!(is_always_active_ffn("model.layers.0.mlp.gate_proj.weight"));
        assert!(is_always_active_ffn(
            "model.layers.7.moe.shared_experts.down_proj.weight"
        ));
        assert!(!is_always_active_ffn(
            "model.layers.7.moe.experts.3.down_proj.weight"
        ));
        assert!(!is_always_active_ffn("model.layers.7.mlp_raon_mhc.phi_pre"));
        assert!(!is_always_active_ffn(
            "model.layers.7.self_attn.q_proj.weight"
        ));
    }

    #[test]
    fn q8f16_and_bf16_always_active_ffn_pass_the_inventory_without_sidecars() {
        let specs = tensor_inventory(&released());
        for qt in [QT_Q8F16, QT_BF16] {
            let mut idx = good_index(&specs);
            retype_ffn(&mut idx, qt);
            check_inventory(&specs, &idx).unwrap_or_else(|e| panic!("qt {qt}: {e}"));
        }
    }

    #[test]
    fn unpadded_shared_down_is_refused_for_every_ffn_storage() {
        let specs = tensor_inventory(&released());
        for qt in [QT_Q8F16, QT_BF16] {
            let mut idx = good_index(&specs);
            retype_ffn(&mut idx, qt);
            let e = idx
                .iter_mut()
                .find(|e| e.name == "model.layers.1.moe.shared_experts.down_proj.weight")
                .unwrap();
            e.shape = vec![2560, 1088];
            e.data_size = payload_bytes(qt, 2560, 1088).unwrap();
            let err = check_inventory(&specs, &idx).unwrap_err();
            assert!(err.contains("unpadded K"), "{err}");
        }
    }

    #[test]
    fn bf16_ffn_images_are_exact_f16_and_the_q8_copy_matches_the_quantizer() {
        let (m, k) = (2, 64);
        let vals: Vec<f32> = (0..m * k)
            .map(|i| {
                let v = ((i as f32) * 0.37 - 20.0) * 0.01;
                f32::from_bits(v.to_bits() & 0xFFFF_0000)
            })
            .collect();
        let data: Vec<u8> = vals
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let (f16, q8, inexact) = bf16_ffn_images(&data, m, k, "t").unwrap();
        assert_eq!(inexact, 0);
        for (i, v) in vals.iter().enumerate() {
            let h = u16::from_le_bytes([f16[2 * i], f16[2 * i + 1]]);
            assert_eq!(f16_to_f32(h).to_bits(), v.to_bits(), "element {i}");
        }
        assert_eq!(q8.len(), m * (k / 32) * 34);
        // First block: scale = amax/127 as f16, int8 round-to-nearest.
        let amax = vals[..32].iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = u16::from_le_bytes([q8[0], q8[1]]);
        assert_eq!(scale, f32_to_f16(amax / 127.0));
        for j in 0..32 {
            let want = (vals[j] * (127.0 / amax)).round() as i8;
            assert!((q8[2 + j] as i8 - want).abs() <= 1, "q[{j}]");
        }
        assert!(bf16_ffn_images(&data, m, 48, "t").is_err());
    }

    #[test]
    fn sidecar_name_follows_repository_convention() {
        assert_eq!(
            awq_sidecar_name("model.layers.1.moe.experts.0.down_proj.weight"),
            "model.layers.1.moe.experts.0.down_proj.awq_scale.weight"
        );
    }
}
