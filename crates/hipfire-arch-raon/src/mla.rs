// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Raon MLA (NoPE, q576/kv512+64) binding onto the generic `LatentAttentionOp`.
//!
//! Math (PLAN §1.6, reference `modeling_raon1_mini.py:381-469`):
//!
//! ```text
//! qlat = RMS576(Wq_a x, eps 1e-6)               q = Wq_b qlat  -> [40, 128+64]
//! [kv512, kpe64] = Wkv_a x ;  c = RMS512(kv512) (the 64 tail channels are NOT normalized)
//! cache row = bf16(c) || kpe64                  ([T, 576] per MLA layer, BF16)
//! [knope128, v128] per head = Wkv_b c ;  key_h = knope_h || kpe (shared over heads)
//! y_h = softmax(q_h . key_h / sqrt(192)) v_h    (no RoPE of any kind, causal j <= prefix+i)
//! y = y * sigmoid(Wg x)  (sigmoid AND product in BF16) ;  o = Wo y
//! ```
//!
//! This file owns only the Raon constants, checkpoint tensor naming, the
//! mirror-admission arithmetic and the stage sequencing. The five
//! projections (`q_a`, `q_b`, `kv_a`, `g`, `o`) are ordinary projections owned
//! by `projection.rs`; the caller runs them and injects `q_b` through a
//! closure because it sits between the latent norm and the attention.

use hipfire_dispatch::pipeline::latent_attention::{
    execute_latent_attention, AbsorbOperands, AttentionRoute, ExpandedMirror, KvExpansion,
    LatentAttentionOp, LatentCache, LatentGeometry, RowRange,
};
use hipfire_dispatch::types::DispatchError;
use rdna_compute::latent_attention_absorb::latent_attention_absorb_packed_elems;
use rdna_compute::latent_attention_decode::latent_attention_decode_workspace_elems;
use rdna_compute::{DType, Gpu, GpuTensor};

use crate::config::RaonConfig;
use crate::projection::{project, OutRound};
use crate::weights::{RaonMixer, RaonWeights};

/// Released geometry (config.json `q_lora_rank`, `kv_lora_rank`, `qk_*`, `v_head_dim`).
pub const RAON_MLA_HEADS: usize = 40;
pub const RAON_MLA_Q_LORA: usize = 576;
pub const RAON_MLA_KV_LATENT: usize = 512;
/// The 64 "rope" channels: plain unrotated shared key/query channels.
pub const RAON_MLA_TAIL: usize = 64;
pub const RAON_MLA_NOPE: usize = 128;
pub const RAON_MLA_V: usize = 128;
/// `mla_latent_norm_eps`.
pub const RAON_MLA_EPS: f32 = 1e-6;

/// Zero-based MLA layer indices of the released checkpoint (config lists are
/// one-based: `[4,8,12,16,20,24,25]`).
pub const RAON_MLA_LAYERS: [usize; 7] = [3, 7, 11, 15, 19, 23, 24];

/// Generic geometry carrying the released Raon values.
pub const fn raon_mla_geometry() -> LatentGeometry {
    LatentGeometry {
        n_heads: RAON_MLA_HEADS,
        nope: RAON_MLA_NOPE,
        tail: RAON_MLA_TAIL,
        v_dim: RAON_MLA_V,
        latent: RAON_MLA_KV_LATENT,
    }
}

/// Score scale `(nope+tail)^-0.5` = `192^-0.5` (never `1/sqrt(576)`).
pub fn raon_mla_scale(g: &LatentGeometry) -> f32 {
    g.default_scale()
}

/// Checkpoint tensor names of one MLA layer's attention mixer
/// (`model.safetensors.index.json`). Shapes: `q_a [576,2560]`, `q_b [7680,576]`,
/// `kv_a [576,2560]`, `kv_b [10240,512]`, `g [5120,2560]`, `o [2560,5120]`,
/// norms `[576]` / `[512]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MlaTensorNames {
    pub q_a: String,
    pub q_a_norm: String,
    pub q_b: String,
    pub kv_a: String,
    pub kv_a_norm: String,
    pub kv_b: String,
    pub g: String,
    pub o: String,
}

impl MlaTensorNames {
    pub fn for_layer(layer: usize) -> Self {
        let p = |s: &str| format!("model.layers.{layer}.self_attn.{s}.weight");
        Self {
            q_a: p("q_a_proj"),
            q_a_norm: p("q_a_layernorm"),
            q_b: p("q_b_proj"),
            kv_a: p("kv_a_proj_with_mqa"),
            kv_a_norm: p("kv_a_layernorm"),
            kv_b: p("kv_b_proj"),
            g: p("g_proj"),
            o: p("o_proj"),
        }
    }
}

/// Persistent per-layer operands of the latent route (weights borrowed).
pub struct RaonMlaWeights<'a> {
    /// `[576]` q latent norm gamma (F32 or BF16).
    pub q_a_norm: &'a GpuTensor,
    /// `[512]` kv latent norm gamma (F32 or BF16).
    pub kv_a_norm: &'a GpuTensor,
    /// How cached latents expand to K_nope/V (dense kernel or external projector).
    pub kv_b: KvExpansion<'a>,
}

/// Per-layer, per-slot mutable operands (all caller-allocated, bounded).
pub struct RaonMlaState<'a> {
    pub cache: LatentCache<'a>,
    pub mirror: ExpandedMirror<'a>,
}

/// Scratch planes for one call of `rows` rows (F32, BF16-valued).
pub struct RaonMlaScratch<'a> {
    /// `[rows, 576]` normalized q latent (input of the `q_b` projection).
    pub qlat: &'a GpuTensor,
    /// `[rows, 40*192]` output of the `q_b` projection.
    pub q: &'a GpuTensor,
    /// `[rows, 40*128]` attention output (gated in place).
    pub y: &'a GpuTensor,
}

// ───────────────────────── route policy ─────────────────────────

/// Load-time MLA route controls (PLAN §3.6). Read once; never re-read per call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MlaPolicy {
    /// `HIPFIRE_RAON_ATTN_TILED` (default 1): tiled causal prefill and
    /// split-KV decode kernels. `0` selects the serial reference kernel.
    pub tiled: bool,
    /// `HIPFIRE_RAON_MLA_ABSORB` (default 1): attend over the latent cache
    /// (absorbed algebra) once a context exceeds the expanded-mirror budget,
    /// instead of refusing it. Not bit-exact to the decompressed route; its
    /// KLD vs the BF16 teacher matched the decompressed route (code24 0.0610
    /// vs 0.0612, WT2 0.2016 vs 0.2018, gfx1201). `0` refuses past the budget.
    pub absorb: bool,
    /// `HIPFIRE_RAON_MIRROR_Q8` (default 1): allow the Q8 expanded K_nope/V mirror
    /// (INT8 codes with one F16 scale per (row, head, plane); about half the BF16
    /// bytes per token, so twice the rows fit the mirror budget). Capacity-based:
    /// `RaonState::fit_mirror_budget` keeps the BF16 mirror when its budget covers
    /// the whole reserved context (it is 1-2% faster) and picks Q8 otherwise.
    /// Needs the tiled kernels (the reference route reads BF16 mirrors only), so
    /// this is forced off when `tiled` is off. `0` always keeps the BF16 mirror.
    pub mirror_q8: bool,
}

/// Which attention operands serve a call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MlaRoute {
    /// Expanded K/V mirror (tiled or reference kernels per [`MlaPolicy`]).
    Decompressed,
    /// Latent-cache attention with the expansion weight absorbed.
    Absorbed,
}

fn parse_flag(name: &str, default: bool) -> Result<bool, String> {
    match hipfire_config::developer_var(name) {
        Err(_) => Ok(default),
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "" => Ok(default),
            "1" | "true" | "on" => Ok(true),
            "0" | "false" | "off" => Ok(false),
            other => Err(format!("{name}={other:?}: expected 0 or 1")),
        },
    }
}

impl MlaPolicy {
    pub const DEFAULT: Self = Self {
        tiled: true,
        absorb: true,
        mirror_q8: true,
    };

    pub fn from_env() -> Result<Self, String> {
        let tiled = parse_flag("HIPFIRE_RAON_ATTN_TILED", Self::DEFAULT.tiled)?;
        Ok(Self {
            tiled,
            absorb: parse_flag("HIPFIRE_RAON_MLA_ABSORB", Self::DEFAULT.absorb)?,
            mirror_q8: parse_flag("HIPFIRE_RAON_MIRROR_Q8", Self::DEFAULT.mirror_q8)? && tiled,
        })
    }

    /// Route for a call that leaves `end` rows cached, given the mirror's
    /// admission budget in rows. Pure, host-side, independent of the
    /// current device length for every `end` within the budget, so a decode
    /// graph's route changes at most once (when the context crosses it).
    pub fn route(&self, end: usize, mirror_budget_rows: usize) -> MlaRoute {
        if self.absorb && end > mirror_budget_rows {
            MlaRoute::Absorbed
        } else {
            MlaRoute::Decompressed
        }
    }
}

/// Per-bundle MLA work buffers: the split-KV decode workspace and, when the
/// absorbed route is enabled, its scratch planes and packed expansion weights
/// (one per MLA layer, built once at load).
pub struct RaonMlaWork {
    policy: MlaPolicy,
    split_ws: GpuTensor,
    absorb: Option<AbsorbWork>,
}

struct AbsorbWork {
    q_abs: GpuTensor,
    z: GpuTensor,
    packed: Vec<GpuTensor>,
}

impl RaonMlaWork {
    /// Allocate for at most `rows` rows per call. Reads the route controls.
    pub fn new(gpu: &mut Gpu, cfg: &RaonConfig, rows: usize) -> Result<Self, String> {
        Self::with_policy(gpu, cfg, rows, MlaPolicy::from_env()?)
    }

    /// Device bytes the work holds for `rows` rows once [`Self::pack_absorbed`]
    /// has run: split-KV workspace, absorbed-route scratch planes and the packed
    /// expansion weights (one per MLA layer). Reads the route controls like
    /// [`Self::new`].
    pub fn required_bytes(cfg: &RaonConfig, rows: usize) -> Result<usize, String> {
        let policy = MlaPolicy::from_env()?;
        let v_width = if policy.absorb {
            cfg.kv_lora.max(cfg.v_head)
        } else {
            cfg.v_head
        };
        let mut bytes =
            (latent_attention_decode_workspace_elems(cfg.n_heads, v_width) * 4).max(256);
        if policy.absorb {
            bytes += (rows * cfg.n_heads * (cfg.kv_lora + cfg.qk_rope) * 4).max(256);
            bytes += (rows * cfg.n_heads * cfg.kv_lora * 4).max(256);
            let packed = latent_attention_absorb_packed_elems(
                cfg.n_heads,
                cfg.qk_nope,
                cfg.v_head,
                cfg.kv_lora,
            );
            bytes += cfg.mla_layers().len() * (packed * 2).max(256);
        }
        Ok(bytes)
    }

    pub fn with_policy(
        gpu: &mut Gpu,
        cfg: &RaonConfig,
        rows: usize,
        policy: MlaPolicy,
    ) -> Result<Self, String> {
        let hip = |e: hip_bridge::HipError| format!("raon mla work: {e:?}");
        let v_width = if policy.absorb {
            cfg.kv_lora.max(cfg.v_head)
        } else {
            cfg.v_head
        };
        let ws = latent_attention_decode_workspace_elems(cfg.n_heads, v_width);
        let split_ws = gpu.zeros(&[ws], DType::F32).map_err(hip)?;
        let absorb = if policy.absorb {
            let q = gpu
                .zeros(&[rows, cfg.n_heads, cfg.kv_lora + cfg.qk_rope], DType::F32)
                .map_err(hip);
            let z = q.and_then(|q| {
                gpu.zeros(&[rows, cfg.n_heads, cfg.kv_lora], DType::F32)
                    .map(|z| (q, z))
                    .map_err(hip)
            });
            match z {
                Ok((q_abs, z)) => Some(AbsorbWork {
                    q_abs,
                    z,
                    packed: Vec::new(),
                }),
                Err(e) => {
                    let _ = gpu.free_tensor(split_ws);
                    return Err(e);
                }
            }
        } else {
            None
        };
        Ok(Self {
            policy,
            split_ws,
            absorb,
        })
    }

    pub fn policy(&self) -> MlaPolicy {
        self.policy
    }

    /// Whether the absorbed route has its packed weights (always true when the
    /// policy enables it and [`Self::pack_absorbed`] succeeded).
    pub fn absorbed_ready(&self) -> bool {
        self.absorb.as_ref().is_some_and(|a| !a.packed.is_empty())
    }

    /// Build the absorbed expansion weights of every MLA layer: run each
    /// layer's own (quantized, basis-rotated) `kv_b` projector on the identity
    /// plane, so the packed weight is exactly the operator the decompressed
    /// route applies, then pack it. Outside capture; once, at load.
    pub fn pack_absorbed(
        &mut self,
        gpu: &mut Gpu,
        cfg: &RaonConfig,
        w: &RaonWeights,
    ) -> Result<(), String> {
        let Some(ab) = self.absorb.as_mut() else {
            return Ok(());
        };
        if !ab.packed.is_empty() {
            return Ok(());
        }
        let hip = |e: hip_bridge::HipError| format!("raon mla absorb pack: {e:?}");
        let (latent, n_out) = (cfg.kv_lora, cfg.n_heads * (cfg.qk_nope + cfg.v_head));
        let mut eye = vec![0f32; latent * latent];
        for c in 0..latent {
            eye[c * latent + c] = 1.0;
        }
        let ident = gpu.upload_f32(&eye, &[latent, latent]).map_err(hip)?;
        let plane = match gpu.zeros(&[latent, n_out], DType::F32) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.free_tensor(ident);
                return Err(hip(e));
            }
        };
        let elems =
            latent_attention_absorb_packed_elems(cfg.n_heads, cfg.qk_nope, cfg.v_head, latent);
        let mut built: Vec<GpuTensor> = Vec::new();
        let r = (|| -> Result<(), String> {
            for layer_idx in cfg.mla_layers() {
                let RaonMixer::Mla(mw) = &w.layers[layer_idx].mixer else {
                    return Err(format!("layer {layer_idx} is not an MLA layer"));
                };
                project(gpu, &mw.kv_b_proj, &ident, &plane, latent, OutRound::Exact)?;
                let packed = gpu.zeros(&[elems], DType::BF16).map_err(hip)?;
                if let Err(e) = gpu.latent_attention_absorb_pack(
                    &plane,
                    &packed,
                    cfg.n_heads,
                    cfg.qk_nope,
                    cfg.v_head,
                    latent,
                ) {
                    let _ = gpu.free_tensor(packed);
                    return Err(hip(e));
                }
                built.push(packed);
            }
            Ok(())
        })();
        let _ = gpu.free_tensor(ident);
        let _ = gpu.free_tensor(plane);
        match r {
            Ok(()) => {
                ab.packed = built;
                Ok(())
            }
            Err(e) => {
                for t in built {
                    let _ = gpu.free_tensor(t);
                }
                Err(e)
            }
        }
    }

    /// Attention route operands for MLA layer ordinal `mi` under `route`.
    pub fn attention_route(
        &self,
        route: MlaRoute,
        mi: usize,
    ) -> Result<AttentionRoute<'_>, String> {
        let policy = self.policy;
        let ws = Some(&self.split_ws);
        match (route, policy.tiled) {
            (MlaRoute::Decompressed, false) => Ok(AttentionRoute::Reference),
            (MlaRoute::Decompressed, true) => Ok(AttentionRoute::Tiled { workspace: ws }),
            (MlaRoute::Absorbed, _) => {
                let ab = self.absorb.as_ref().filter(|a| mi < a.packed.len()).ok_or(
                    "raon mla: absorbed route selected but its packed weights are not built",
                )?;
                Ok(AttentionRoute::Absorbed {
                    ops: AbsorbOperands {
                        packed: &ab.packed[mi],
                        q_abs: &ab.q_abs,
                        z: &ab.z,
                    },
                    workspace: ws,
                })
            }
        }
    }

    /// Every device address a captured graph bakes from this owner.
    pub fn binding_key(&self) -> Vec<usize> {
        let mut k = vec![
            self.split_ws.buf.as_ptr() as usize,
            self.policy.tiled as usize,
            self.policy.absorb as usize,
        ];
        if let Some(a) = &self.absorb {
            k.push(a.q_abs.buf.as_ptr() as usize);
            k.push(a.z.buf.as_ptr() as usize);
            k.extend(a.packed.iter().map(|t| t.buf.as_ptr() as usize));
        }
        k
    }

    pub fn free(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.split_ws);
        if let Some(a) = self.absorb {
            let _ = gpu.free_tensor(a.q_abs);
            let _ = gpu.free_tensor(a.z);
            for t in a.packed {
                let _ = gpu.free_tensor(t);
            }
        }
    }
}

/// Bytes of the BF16 latent cache for `rows` rows of one layer/slot.
pub const fn latent_cache_bytes(rows: usize) -> usize {
    rows * (RAON_MLA_KV_LATENT + RAON_MLA_TAIL) * 2
}

/// Bytes per token row of the expanded BF16 K_nope+V mirror (all heads).
pub const fn mirror_bytes_per_row() -> usize {
    RAON_MLA_HEADS * (RAON_MLA_NOPE + RAON_MLA_V) * 2
}

/// Rows of expanded mirror to admit for a layer/slot: `needed_rows` when it
/// fits `budget_bytes`, otherwise `None` (route not admitted; the caller keeps
/// latent-only storage and must use a tiled/absorbed route). Never sized from
/// the model's maximum sequence unconditionally.
pub fn admitted_mirror_rows(needed_rows: usize, budget_bytes: usize) -> Option<usize> {
    let need = needed_rows.checked_mul(mirror_bytes_per_row())?;
    (need <= budget_bytes).then_some(needed_rows)
}

/// Run the Raon MLA core for `rows` rows starting at the device position held
/// in `state.cache.pos`:
///
/// 1. `qlat = RMS576(q_a_out)`; 2. `project_q_b(gpu, qlat, q)`;
/// 3. split `kv_a_out` into the latent cache (norm over 512 only);
/// 4. decompress the new rows into the mirror; 5. serial attention; 6. BF16
///    sigmoid gate with `g_out`.
///
/// On return `scratch.y` holds the gated `[rows, 5120]` context, ready for the
/// `o` projection. The position scalar is not advanced here (the layer
/// schedule owner advances it once per call after all MLA layers consumed it).
pub fn raon_mla_core(
    gpu: &mut Gpu,
    w: &RaonMlaWeights<'_>,
    st: &RaonMlaState<'_>,
    scratch: &RaonMlaScratch<'_>,
    q_a_out: &GpuTensor,
    kv_a_out: &GpuTensor,
    g_out: &GpuTensor,
    rows: usize,
    project_q_b: &mut dyn FnMut(&mut Gpu, &GpuTensor, &GpuTensor) -> Result<(), DispatchError>,
) -> Result<(), DispatchError> {
    let geometry = raon_mla_geometry();
    execute_latent_attention(
        gpu,
        &LatentAttentionOp::NormRows {
            x: q_a_out,
            gamma: w.q_a_norm,
            out: scratch.qlat,
            rows,
            width: RAON_MLA_Q_LORA,
            eps: RAON_MLA_EPS,
        },
    )?;
    project_q_b(gpu, scratch.qlat, scratch.q)?;
    execute_latent_attention(
        gpu,
        &LatentAttentionOp::SplitKv {
            geometry,
            cache: st.cache,
            kv_a: kv_a_out,
            gamma: w.kv_a_norm,
            rows,
            eps: RAON_MLA_EPS,
        },
    )?;
    match w.kv_b {
        KvExpansion::Dense { .. } => {
            execute_latent_attention(
                gpu,
                &LatentAttentionOp::Decompress {
                    geometry,
                    cache: st.cache,
                    mirror: st.mirror,
                    expansion: w.kv_b,
                    range: RowRange::NewRows { count: rows },
                },
            )?;
        }
        KvExpansion::External { .. } => {
            return Err(DispatchError::Hip(
                "raon_mla_core: external kv_b projector needs the caller to run \
                 GatherLatent -> projector -> Decompress(External) itself"
                    .into(),
            ));
        }
    }
    execute_latent_attention(
        gpu,
        &LatentAttentionOp::Attention {
            geometry,
            cache: st.cache,
            mirror: Some(st.mirror),
            q: scratch.q,
            y: scratch.y,
            rows,
            scale: raon_mla_scale(&geometry),
            route: AttentionRoute::Reference,
        },
    )?;
    execute_latent_attention(
        gpu,
        &LatentAttentionOp::Gate {
            y: scratch.y,
            g: g_out,
            out: scratch.y,
            elements: rows * RAON_MLA_HEADS * RAON_MLA_V,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_match_checkpoint_index() {
        let n = MlaTensorNames::for_layer(3);
        assert_eq!(n.q_a, "model.layers.3.self_attn.q_a_proj.weight");
        assert_eq!(n.kv_a, "model.layers.3.self_attn.kv_a_proj_with_mqa.weight");
        assert_eq!(
            n.kv_a_norm,
            "model.layers.3.self_attn.kv_a_layernorm.weight"
        );
        assert_eq!(n.kv_b, "model.layers.3.self_attn.kv_b_proj.weight");
        assert_eq!(n.g, "model.layers.3.self_attn.g_proj.weight");
        assert_eq!(n.o, "model.layers.3.self_attn.o_proj.weight");
    }

    #[test]
    fn geometry_is_192_128_with_192_scale() {
        let g = raon_mla_geometry();
        assert_eq!(g.qk_dim(), 192);
        assert_eq!(g.v_dim, 128);
        assert_eq!(g.row_width(), 576);
        assert_eq!(g.expanded_width(), 10240);
        let s = raon_mla_scale(&g);
        assert!((s - 192f32.powf(-0.5)).abs() < 1e-9);
        assert!((s - 576f32.powf(-0.5)).abs() > 1e-3);
    }

    #[test]
    fn mirror_admission_is_bounded() {
        assert_eq!(mirror_bytes_per_row(), 40 * 256 * 2);
        assert_eq!(latent_cache_bytes(1), 1152);
        assert_eq!(
            admitted_mirror_rows(1024, 1024 * mirror_bytes_per_row()),
            Some(1024)
        );
        assert_eq!(
            admitted_mirror_rows(1025, 1024 * mirror_bytes_per_row()),
            None
        );
        assert_eq!(admitted_mirror_rows(usize::MAX, usize::MAX), None);
    }
}
