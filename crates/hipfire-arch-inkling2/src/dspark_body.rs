// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU body of Inkling-2-Mini's bundled DSpark draft (`mtp.safetensors` at
//! `thinkingmachines/Inkling-2-Mini@c9d02830`), bound to the shared DSpark
//! driver as a [`DsparkCommittedBody`].
//!
//! It reproduces the CPU reference (`hipfire-loader/examples/inkling_mtp/
//! reference.rs`, written from vLLM `b7a65380` `nvidia/dspark.py`) with
//! existing typed HIP operators only. Every tensor the reference stores in
//! BF16 is rounded to BF16 here at the same boundary (F32 math in between);
//! dot-product accumulation order is the operators', so the body matches the
//! reference within pinned tolerances, not bit for bit.
//!
//! * Committed context: each committed row's four taps (the streams entering
//!   visits 5, 23 and 41 of the looped schedule, then the final normalized
//!   state) are rounded to BF16, mixed with `bf16(softmax(fuse_weight[l]))`
//!   (F32, tap order), normalized by draft layer `l`'s `dflare_norm` and
//!   projected by the separate `k_proj_ctx` / `v_proj_ctx`; K gets
//!   `k_norm_ctx` and NeoX RoPE at the row's absolute position. Draft layers
//!   0 and 2 keep at least the last 1024 + 8 rows, layer 1 the whole prefix.
//! * Block: row 0 is the anchor's target embedding through the target
//!   `embed_norm`, rows 1..8 are `mask_embed`, at positions `P..P+8`; three
//!   pre-norm layers attend bidirectionally to the block plus the committed
//!   context before `P` (layers 0 and 2: the last 1024 positions before `P`).
//! * Head: base logits are the target unembed of fill-in rows `1..=k`; the
//!   recurrent Markov head starts from a zero state with the anchor as the
//!   previous id and `f_rnn` from the taps of row `P - 1`; `hidden_out` is
//!   folded into the unembed ([`InklingFold`]); logits are
//!   `bf16(30·tanh(bf16(base + bias)/30))` over the 200058 visible ids and the
//!   greedy pick is the first maximum.

use crate::dspark_weights::{InklingDsparkConfig, InklingDsparkWeights, FOLD_BIAS, FOLD_WEIGHT};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::gemm::{GemmFamily, GemmParams};
use hipfire_dispatch::families::gemv::WeightRef;
use hipfire_dispatch::types::{DispatchError, KernelKey};
use hipfire_runtime::dspark_core::{DraftResult, DsparkCommittedBody, DsparkTapSchema};
use rayon::prelude::*;
use rdna_compute::{DType, Gpu, GpuTensor};

pub const HIDDEN: usize = 2304;
pub const N_HEAD: usize = 24;
pub const N_KV: usize = 8;
pub const HEAD_DIM: usize = 128;
pub const FFN: usize = 9216;
pub const N_LAYER: usize = 3;
/// Drafted block rows: the anchor plus seven fill-in rows.
pub const BLOCK: usize = 8;
/// Proposals per window.
pub const K_MAX: usize = BLOCK - 1;
pub const RANK: usize = 256;
pub const N_TAP: usize = 4;
pub const LOCAL_WINDOW: usize = 1024;
pub const ROPE_THETA: f32 = 1_000_000.0;
pub const EPS: f32 = 1e-6;
pub const SOFTCAP: f32 = 30.0;
/// Visible (unpadded) vocabulary.
pub const VOCAB: usize = 200_058;
/// Padded vocabulary (`markov_w1` rows, target embedding/unembed rows).
pub const PADDED_VOCAB: usize = 201_024;
/// Ordered tap ids: the streams entering visits 5, 23 and 41 (0-based, loop
/// passes counted), then 48 = the final normalized state.
pub const TAP_VISITS: [usize; N_TAP] = [5, 23, 41, 48];
/// F32 values per committed tap row.
pub const TAP_ROW: usize = N_TAP * HIDDEN;
const LOCAL_LAYERS: [bool; N_LAYER] = [true, false, true];
const QD: usize = N_HEAD * HEAD_DIM;
const KVD: usize = N_KV * HEAD_DIM;
/// Rows a local layer keeps on compaction: its window plus one truncatable window.
const LOCAL_KEEP: usize = LOCAL_WINDOW + BLOCK;
/// Committed rows processed per ingest pass.
const INGEST_ROWS: usize = 256;
/// Last committed tap rows kept (f_rnn of the anchor, terminal truncation).
const TAP_RING: usize = 2 * BLOCK;
/// Global-layer context rows allocated up front (grown by doubling).
const GLOBAL_INITIAL_ROWS: usize = 4096;

/// FP32 → BF16 bits, round to nearest even (the reference's `.to(bf16)`).
pub fn bf16_bits(x: f32) -> u16 {
    if x.is_nan() {
        return (x.to_bits() >> 16) as u16 | 0x40;
    }
    let b = x.to_bits();
    (b.wrapping_add(0x7FFF + ((b >> 16) & 1)) >> 16) as u16
}

fn bf16_from_f64(x: f64) -> u16 {
    let f = x as f32;
    if !x.is_finite() || f as f64 == x {
        return bf16_bits(f);
    }
    // Round to odd in F32 first, then RNE: one rounding of the F64 value.
    let mut b = f.to_bits();
    if (f as f64).abs() > x.abs() {
        b -= 1;
    }
    bf16_bits(f32::from_bits(b | 1))
}

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn hip<E: std::fmt::Debug>(what: &'static str) -> impl FnOnce(E) -> String {
    move |err| format!("inkling dspark: {what}: {err:?}")
}

/// Non-owning `shape` view of `t` starting `off` elements in.
fn view(t: &GpuTensor, off: usize, shape: &[usize]) -> GpuTensor {
    let es = t.dtype.size();
    let n: usize = shape.iter().product();
    GpuTensor {
        buf: t.buf.byte_view(off * es, n * es),
        shape: shape.to_vec(),
        dtype: t.dtype,
    }
}

fn round(gpu: &mut Gpu, t: &GpuTensor, n: usize) -> Result<(), String> {
    gpu.ffn_round_bf16_f32(t, n).map_err(hip("bf16 round"))
}

/// Typed dispatch of the draft's BF16-weight linears
/// (`KernelKey::GemmBf16XF32Batched`): `y[n, m] = x[n, k] · w[m, k]ᵀ`, F32
/// activations and output, F32 accumulation.
struct Lin {
    family: GemmFamily,
    ctx: DispatchCtx,
}

impl Lin {
    fn new(gpu: &Gpu) -> Self {
        Self {
            family: GemmFamily::new(),
            ctx: DispatchCtx::new(gpu),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        gpu: &mut Gpu,
        w: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(), DispatchError> {
        let w = WeightRef {
            buf: w,
            dtype: DType::BF16,
            m,
            k,
            row_stride: 0,
            awq_scale: None,
            lloyd_lut_e4m3: None,
            lloyd_lut_f16: None,
            lloyd_lut_c16: None,
        };
        let params = GemmParams {
            w: &w,
            x,
            y,
            batch_size: n,
        };
        self.family
            .run_key(KernelKey::GemmBf16XF32Batched, &self.ctx, gpu, &params)
    }
}

/// `dst[dst_off..+n] = src[src_off..+n]` (F32 elements), stream ordered.
fn copy(
    gpu: &Gpu,
    dst: &GpuTensor,
    dst_off: usize,
    src: &GpuTensor,
    src_off: usize,
    n: usize,
) -> Result<(), String> {
    gpu.memcpy_dtod_at_auto(&dst.buf, dst_off * 4, &src.buf, src_off * 4, n * 4)
        .map_err(hip("d2d copy"))
}

fn upload_i32(gpu: &Gpu, dst: &GpuTensor, vals: &[i32]) -> Result<(), String> {
    let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    gpu.memcpy_htod_auto(&dst.buf.byte_view(0, bytes.len()), &bytes)
        .map_err(hip("upload i32"))
}

fn alloc(gpu: &mut Gpu, n: usize) -> Result<GpuTensor, String> {
    gpu.alloc_tensor(&[n], DType::F32).map_err(hip("alloc"))
}

fn upload_bf16(gpu: &mut Gpu, bytes: &[u8], shape: &[usize]) -> Result<GpuTensor, String> {
    let t = gpu.zeros(shape, DType::BF16).map_err(hip("alloc bf16"))?;
    gpu.memcpy_htod_auto(&t.buf, bytes)
        .map_err(hip("upload bf16"))?;
    Ok(t)
}

fn upload_f32(gpu: &mut Gpu, v: &[f32]) -> Result<GpuTensor, String> {
    gpu.upload_f32(v, &[v.len()]).map_err(hip("upload f32"))
}

fn widen(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

/// Raw BF16 bytes of `name` in a validated sidecar ([`InklingDsparkWeights::load`]
/// already checked the exact tensor inventory, dtypes and shapes).
fn sidecar_tensor<'a>(w: &'a InklingDsparkWeights, name: &str) -> Result<&'a [u8], String> {
    w.tensors
        .iter()
        .find(|(n, _)| n == name)
        .map(|(_, t)| t.data.as_slice())
        .ok_or_else(|| format!("inkling dspark: the sidecar has no {name}"))
}

/// The sidecar describes exactly the draft this body implements.
fn check_config(c: &InklingDsparkConfig) -> Result<(), String> {
    let ok = c.hidden_size == HIDDEN
        && c.num_hidden_layers == N_LAYER
        && c.num_attention_heads == N_HEAD
        && c.num_key_value_heads == N_KV
        && c.head_dim == HEAD_DIM
        && c.intermediate_size == FFN
        && c.block_size == BLOCK
        && c.proposal_capacity == K_MAX
        && c.markov_rank == RANK
        && c.markov_head_type == "rnn_hidden"
        && c.target_layer_ids == TAP_VISITS
        && c.local_layer_ids == [0, 2]
        && c.sliding_window_size == LOCAL_WINDOW
        && c.vocab_size == PADDED_VOCAB
        && c.unpadded_vocab_size == VOCAB
        && c.rms_norm_eps as f32 == EPS
        && c.rope_theta as f32 == ROPE_THETA
        && c.final_logit_softcapping as f32 == SOFTCAP;
    if ok {
        Ok(())
    } else {
        Err(format!(
            "inkling dspark: sidecar geometry {c:?} is not the v1 body's"
        ))
    }
}

/// `hidden_out` folded into the shared unembed: `w[v] = bf16(U[v] · W_hidden_out)`
/// (`[VOCAB, RANK]` BF16 bits) and `b[v] = bf16(U[v] · b_hidden_out)` (BF16-valued
/// F32), accumulated in F64 with one rounding, for the visible vocabulary.
pub struct InklingFold {
    pub w_bf16: Vec<u16>,
    pub b: Vec<f32>,
}

impl InklingFold {
    /// The fold the sidecar carries (`markov_head.hidden_out_fold.*`, derived
    /// by the packer from the paired trunk's effective unembed).
    pub fn from_sidecar(w: &InklingDsparkWeights) -> Result<Self, String> {
        let fw = sidecar_tensor(w, FOLD_WEIGHT)?;
        let fb = sidecar_tensor(w, FOLD_BIAS)?;
        if fw.len() < VOCAB * RANK * 2 || fb.len() < VOCAB * 2 {
            return Err("inkling dspark: sidecar fold shorter than the visible vocabulary".into());
        }
        Ok(Self {
            w_bf16: fw[..VOCAB * RANK * 2]
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect(),
            b: widen(&fb[..VOCAB * 2]),
        })
    }
}

impl InklingFold {
    /// Fold from the target's effective unembed rows (`unembed_row(v, out)`
    /// writes row `v`, `HIDDEN` F32) and the draft's `hidden_out`
    /// (`[HIDDEN, RANK]` weight, `[HIDDEN]` bias, F32 widenings).
    pub fn compute(
        unembed_row: &(dyn Fn(usize, &mut [f32]) + Sync),
        hidden_out_w: &[f32],
        hidden_out_b: &[f32],
    ) -> Result<Self, String> {
        if hidden_out_w.len() != HIDDEN * RANK || hidden_out_b.len() != HIDDEN {
            return Err("inkling dspark fold: hidden_out shape".into());
        }
        let rows: Vec<(Vec<u16>, f32)> = (0..VOCAB)
            .into_par_iter()
            .map_init(
                || vec![0.0f32; HIDDEN],
                |u, v| {
                    unembed_row(v, u);
                    Self::row(u, hidden_out_w, hidden_out_b)
                },
            )
            .collect();
        let mut w_bf16 = Vec::with_capacity(VOCAB * RANK);
        let mut b = Vec::with_capacity(VOCAB);
        for (w, bias) in rows {
            w_bf16.extend_from_slice(&w);
            b.push(bias);
        }
        Ok(Self { w_bf16, b })
    }

    /// One folded row from unembed row `u` (`HIDDEN`): `(w[RANK] BF16 bits, b)`.
    pub fn row(u: &[f32], hidden_out_w: &[f32], hidden_out_b: &[f32]) -> (Vec<u16>, f32) {
        let mut acc = [0.0f64; RANK];
        let mut b = 0.0f64;
        for (k, &uk) in u.iter().enumerate() {
            let uk = uk as f64;
            for (a, &h) in acc.iter_mut().zip(&hidden_out_w[k * RANK..(k + 1) * RANK]) {
                *a += uk * h as f64;
            }
            b += uk * hidden_out_b[k] as f64;
        }
        (
            acc.iter().map(|&a| bf16_from_f64(a)).collect(),
            bf16_to_f32(bf16_from_f64(b)),
        )
    }
}

/// The target tensors the draft shares: the input embedding through
/// `embed_norm` for the anchor row, and the unembed for the base logits.
pub trait InklingTargetHead {
    /// `out[0..HIDDEN] = bf16(rmsnorm(embed[id]) · embed_norm)` (F32).
    fn anchor_embedding(&mut self, gpu: &mut Gpu, id: u32, out: &GpuTensor) -> Result<(), String>;
    /// `out[r, v] = x[r] · U[v]` (F32) for `rows` rows of `x` (`[rows, HIDDEN]`)
    /// over the visible vocabulary `v < VOCAB`.
    fn unembed(
        &mut self,
        gpu: &mut Gpu,
        x: &GpuTensor,
        rows: usize,
        out: &GpuTensor,
    ) -> Result<(), String>;
    fn free(self: Box<Self>, gpu: &mut Gpu);
}

/// Raw BF16 target embedding / `embed_norm` / unembed, resident on the device
/// (the source teacher's tensors; the CPU reference binds the same ones).
pub struct Bf16TargetHead {
    embed: GpuTensor,
    embed_norm: GpuTensor,
    unembed: GpuTensor,
    id: GpuTensor,
    lin: Lin,
}

impl Bf16TargetHead {
    /// `embed`, `unembed`: `[PADDED_VOCAB, HIDDEN]` BF16 bytes; `embed_norm`:
    /// `[HIDDEN]` BF16 bytes.
    pub fn upload(
        gpu: &mut Gpu,
        embed: &[u8],
        embed_norm: &[u8],
        unembed: &[u8],
    ) -> Result<Self, String> {
        if embed.len() != PADDED_VOCAB * HIDDEN * 2
            || unembed.len() != PADDED_VOCAB * HIDDEN * 2
            || embed_norm.len() != HIDDEN * 2
        {
            return Err("inkling dspark target head: embed/unembed shape".into());
        }
        Ok(Self {
            embed: upload_bf16(gpu, embed, &[PADDED_VOCAB, HIDDEN])?,
            embed_norm: upload_f32(gpu, &widen(embed_norm))?,
            unembed: upload_bf16(gpu, unembed, &[PADDED_VOCAB, HIDDEN])?,
            id: gpu.upload_raw(&[0u8; 4], &[4]).map_err(hip("alloc id"))?,
            lin: Lin::new(gpu),
        })
    }
}

impl InklingTargetHead for Bf16TargetHead {
    fn anchor_embedding(&mut self, gpu: &mut Gpu, id: u32, out: &GpuTensor) -> Result<(), String> {
        if id as usize >= PADDED_VOCAB {
            return Err(format!(
                "inkling dspark: anchor id {id} outside the vocabulary"
            ));
        }
        upload_i32(gpu, &self.id, &[id as i32])?;
        let row = view(out, 0, &[HIDDEN]);
        gpu.embedding_lookup_bf16_batched(&self.embed, &row, &self.id, 1, HIDDEN)
            .map_err(hip("anchor embedding"))?;
        gpu.rmsnorm_batched(&row, &self.embed_norm, &row, 1, HIDDEN, EPS)
            .map_err(hip("anchor embed_norm"))?;
        round(gpu, &row, HIDDEN)
    }

    fn unembed(
        &mut self,
        gpu: &mut Gpu,
        x: &GpuTensor,
        rows: usize,
        out: &GpuTensor,
    ) -> Result<(), String> {
        let u = view(&self.unembed, 0, &[VOCAB, HIDDEN]);
        self.lin
            .run(gpu, &u, x, out, VOCAB, HIDDEN, rows)
            .map_err(hip("unembed"))
    }

    fn free(self: Box<Self>, gpu: &mut Gpu) {
        for t in [self.embed, self.embed_norm, self.unembed, self.id] {
            let _ = gpu.free_tensor(t);
        }
    }
}

// ---------------------------------------------------------------------------
// Body
// ---------------------------------------------------------------------------

struct Layer {
    local: bool,
    dflare_norm: GpuTensor,
    input_ln: GpuTensor,
    post_ln: GpuTensor,
    q_norm: GpuTensor,
    k_norm: GpuTensor,
    k_norm_ctx: GpuTensor,
    wq: GpuTensor,
    wk: GpuTensor,
    wv: GpuTensor,
    wk_ctx: GpuTensor,
    wv_ctx: GpuTensor,
    wo: GpuTensor,
    w_gate: GpuTensor,
    w_up: GpuTensor,
    w_down: GpuTensor,
    /// Context K (normalized, rotated) / V of committed rows `[base, len)` at
    /// slots `row - base`, plus `BLOCK` transient slots after the last row.
    k_ctx: GpuTensor,
    v_ctx: GpuTensor,
    /// Slots (rows) of `k_ctx` / `v_ctx`, transient tail included.
    slots: usize,
    base: usize,
}

impl Layer {
    fn tensors(self) -> [GpuTensor; 17] {
        [
            self.dflare_norm,
            self.input_ln,
            self.post_ln,
            self.q_norm,
            self.k_norm,
            self.k_norm_ctx,
            self.wq,
            self.wk,
            self.wv,
            self.wk_ctx,
            self.wv_ctx,
            self.wo,
            self.w_gate,
            self.w_up,
            self.w_down,
            self.k_ctx,
            self.v_ctx,
        ]
    }
}

/// Reusable activations.
struct Scratch {
    taps: GpuTensor,      // [INGEST_ROWS * TAP_ROW], BF16-rounded copy
    tap_major: GpuTensor, // [N_TAP * INGEST_ROWS * HIDDEN]
    mixed: GpuTensor,     // [INGEST_ROWS * HIDDEN]
    mix_tmp: GpuTensor,   // [INGEST_ROWS * HIDDEN]
    ctx_k: GpuTensor,     // [INGEST_ROWS * KVD]
    ctx_v: GpuTensor,     // [INGEST_ROWS * KVD]
    ctx_pos: GpuTensor,   // [INGEST_ROWS] i32
    x: GpuTensor,         // [BLOCK * HIDDEN]
    residual: GpuTensor,
    sum: GpuTensor,
    h: GpuTensor,
    o: GpuTensor,
    q: GpuTensor,    // [BLOCK * QD]
    attn: GpuTensor, // [BLOCK * QD]
    k: GpuTensor,    // [BLOCK * KVD]
    v: GpuTensor,
    g: GpuTensor, // [BLOCK * FFN]
    u: GpuTensor,
    act: GpuTensor,
    blk_pos: GpuTensor, // [BLOCK] i32
    frnn: GpuTensor,    // [HIDDEN]
    xctx: GpuTensor,    // [K_MAX * 2 * HIDDEN]
    ctx_raw: GpuTensor, // [K_MAX * 3 * RANK]
    sw: GpuTensor,      // [2 * RANK]: state | markov_w1[prev]
    rec: GpuTensor,     // [3 * RANK]
    raw: GpuTensor,     // [3 * RANK]
    keep: GpuTensor,    // [RANK]
    omg: GpuTensor,     // [RANK]
    upd: GpuTensor,     // [RANK]
    ones: GpuTensor,    // [RANK]
    base: GpuTensor,    // [K_MAX * VOCAB]
    bias: GpuTensor,    // [VOCAB]
    logits: GpuTensor,  // [VOCAB]
    prev: GpuTensor,    // Raw [4]
}

impl Scratch {
    fn new(gpu: &mut Gpu) -> Result<Self, String> {
        let ones = upload_f32(gpu, &[1.0f32; RANK])?;
        Ok(Self {
            taps: alloc(gpu, INGEST_ROWS * TAP_ROW)?,
            tap_major: alloc(gpu, N_TAP * INGEST_ROWS * HIDDEN)?,
            mixed: alloc(gpu, INGEST_ROWS * HIDDEN)?,
            mix_tmp: alloc(gpu, INGEST_ROWS * HIDDEN)?,
            ctx_k: alloc(gpu, INGEST_ROWS * KVD)?,
            ctx_v: alloc(gpu, INGEST_ROWS * KVD)?,
            ctx_pos: alloc(gpu, INGEST_ROWS)?,
            x: alloc(gpu, BLOCK * HIDDEN)?,
            residual: alloc(gpu, BLOCK * HIDDEN)?,
            sum: alloc(gpu, BLOCK * HIDDEN)?,
            h: alloc(gpu, BLOCK * HIDDEN)?,
            o: alloc(gpu, BLOCK * HIDDEN)?,
            q: alloc(gpu, BLOCK * QD)?,
            attn: alloc(gpu, BLOCK * QD)?,
            k: alloc(gpu, BLOCK * KVD)?,
            v: alloc(gpu, BLOCK * KVD)?,
            g: alloc(gpu, BLOCK * FFN)?,
            u: alloc(gpu, BLOCK * FFN)?,
            act: alloc(gpu, BLOCK * FFN)?,
            blk_pos: alloc(gpu, BLOCK)?,
            frnn: alloc(gpu, HIDDEN)?,
            xctx: alloc(gpu, K_MAX * 2 * HIDDEN)?,
            ctx_raw: alloc(gpu, K_MAX * 3 * RANK)?,
            sw: alloc(gpu, 2 * RANK)?,
            rec: alloc(gpu, 3 * RANK)?,
            raw: alloc(gpu, 3 * RANK)?,
            keep: alloc(gpu, RANK)?,
            omg: alloc(gpu, RANK)?,
            upd: alloc(gpu, RANK)?,
            ones,
            base: alloc(gpu, K_MAX * VOCAB)?,
            bias: alloc(gpu, VOCAB)?,
            logits: alloc(gpu, VOCAB)?,
            prev: gpu.upload_raw(&[0u8; 4], &[4]).map_err(hip("alloc prev"))?,
        })
    }

    fn tensors(self) -> Vec<GpuTensor> {
        vec![
            self.taps,
            self.tap_major,
            self.mixed,
            self.mix_tmp,
            self.ctx_k,
            self.ctx_v,
            self.ctx_pos,
            self.x,
            self.residual,
            self.sum,
            self.h,
            self.o,
            self.q,
            self.attn,
            self.k,
            self.v,
            self.g,
            self.u,
            self.act,
            self.blk_pos,
            self.frnn,
            self.xctx,
            self.ctx_raw,
            self.sw,
            self.rec,
            self.raw,
            self.keep,
            self.omg,
            self.upd,
            self.ones,
            self.base,
            self.bias,
            self.logits,
            self.prev,
        ]
    }
}

/// What the last [`DsparkCommittedBody::score_block`] read, for state oracles.
#[derive(Clone, Debug, Default)]
pub struct InklingDraftTrace {
    /// Absolute position of the committed row `f_rnn` came from (`P - 1`).
    pub frnn_row: Option<usize>,
    /// The BF16-valued `f_rnn` vector (`[HIDDEN]`), when tracing.
    pub frnn: Vec<f32>,
    /// The Markov state after each step (`[k][RANK]`), when tracing.
    pub states: Vec<Vec<f32>>,
    /// The softcapped BF16-valued logits of each step (`[k][VOCAB]`), when tracing.
    pub logits: Vec<Vec<f32>>,
}

/// The Inkling-2-Mini DSpark draft body (see the module docs).
pub struct InklingDsparkBody {
    schema: DsparkTapSchema,
    layers: Vec<Layer>,
    norm: GpuTensor,
    mask_embed: GpuTensor,
    markov_norm: GpuTensor,
    /// `bf16(softmax(fuse_weight[row]))`; rows 0..3 = draft layers, 3 = Markov.
    alpha: [[f32; N_TAP]; N_LAYER + 1],
    markov_w1: GpuTensor,
    /// `rnn_proj` columns `[2R, 2R + 2H)`: the per-row `[hidden, f_rnn]` term.
    rnn_w_ctx: GpuTensor,
    /// `rnn_proj` columns `[0, 2R)`: the sequential `[state, prev]` term.
    rnn_w_rec: GpuTensor,
    rnn_b: GpuTensor,
    fold_w: GpuTensor,
    fold_b: GpuTensor,
    head: Box<dyn InklingTargetHead>,
    /// BF16-rounded taps of the last `TAP_RING` committed rows, slot `pos % TAP_RING`.
    tap_ring: GpuTensor,
    s: Scratch,
    len: usize,
    position_capacity: usize,
    trace_enabled: bool,
    trace: InklingDraftTrace,
    lin: Lin,
}

impl InklingDsparkBody {
    /// Upload the draft from a validated sidecar, bind the target `head` and
    /// the `fold` derived from that head's unembed ([`InklingFold::from_sidecar`]
    /// for the packed trunk). `position_capacity` bounds the absolute positions
    /// a drafted block may reach (the target's context capacity).
    pub fn load(
        gpu: &mut Gpu,
        src: &InklingDsparkWeights,
        head: Box<dyn InklingTargetHead>,
        fold: &InklingFold,
        position_capacity: usize,
    ) -> Result<Self, String> {
        check_config(&src.cfg)?;
        if fold.w_bf16.len() != VOCAB * RANK || fold.b.len() != VOCAB {
            return Err("inkling dspark: fold shape".into());
        }
        let bytes = |n: &str| sidecar_tensor(src, n);
        let f32w = |gpu: &mut Gpu, n: &str| -> Result<GpuTensor, String> {
            upload_f32(gpu, &widen(bytes(n)?))
        };
        let mut layers = Vec::with_capacity(N_LAYER);
        for (i, &local) in LOCAL_LAYERS.iter().enumerate() {
            let p = |n: &str| format!("layers.{i}.{n}");
            let bw = |gpu: &mut Gpu, n: &str, shape: &[usize]| -> Result<GpuTensor, String> {
                upload_bf16(gpu, bytes(&p(n))?, shape)
            };
            let slots = if local {
                2 * LOCAL_KEEP + INGEST_ROWS + BLOCK
            } else {
                position_capacity.min(GLOBAL_INITIAL_ROWS) + BLOCK
            };
            layers.push(Layer {
                local,
                dflare_norm: f32w(gpu, &p("dflare_norm.weight"))?,
                input_ln: f32w(gpu, &p("input_layernorm.weight"))?,
                post_ln: f32w(gpu, &p("post_attention_layernorm.weight"))?,
                q_norm: f32w(gpu, &p("self_attn.q_norm.weight"))?,
                k_norm: f32w(gpu, &p("self_attn.k_norm.weight"))?,
                k_norm_ctx: f32w(gpu, &p("self_attn.k_norm_ctx.weight"))?,
                wq: bw(gpu, "self_attn.q_proj.weight", &[QD, HIDDEN])?,
                wk: bw(gpu, "self_attn.k_proj.weight", &[KVD, HIDDEN])?,
                wv: bw(gpu, "self_attn.v_proj.weight", &[KVD, HIDDEN])?,
                wk_ctx: bw(gpu, "self_attn.k_proj_ctx.weight", &[KVD, HIDDEN])?,
                wv_ctx: bw(gpu, "self_attn.v_proj_ctx.weight", &[KVD, HIDDEN])?,
                wo: bw(gpu, "self_attn.o_proj.weight", &[HIDDEN, QD])?,
                w_gate: bw(gpu, "mlp.gate_proj.weight", &[FFN, HIDDEN])?,
                w_up: bw(gpu, "mlp.up_proj.weight", &[FFN, HIDDEN])?,
                w_down: bw(gpu, "mlp.down_proj.weight", &[HIDDEN, FFN])?,
                k_ctx: alloc(gpu, slots * KVD)?,
                v_ctx: alloc(gpu, slots * KVD)?,
                slots,
                base: 0,
            });
        }
        let fuse = widen(bytes("fuse_weight")?);
        let mut alpha = [[0.0f32; N_TAP]; N_LAYER + 1];
        for (r, row) in alpha.iter_mut().enumerate() {
            let x = &fuse[r * N_TAP..(r + 1) * N_TAP];
            let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = x.iter().map(|v| (v - m).exp()).collect();
            let sum: f32 = e.iter().sum();
            for (a, ev) in row.iter_mut().zip(&e) {
                *a = bf16_to_f32(bf16_bits(ev / sum));
            }
        }
        // Split rnn_proj [3R, 2R + 2H] by columns into the sequential and the
        // per-row operand.
        let rnn = bytes("markov_head.rnn_proj.weight")?;
        let cols = 2 * RANK + 2 * HIDDEN;
        let mut rec = Vec::with_capacity(3 * RANK * 2 * RANK * 2);
        let mut ctx = Vec::with_capacity(3 * RANK * 2 * HIDDEN * 2);
        for r in 0..3 * RANK {
            let row = &rnn[r * cols * 2..(r + 1) * cols * 2];
            rec.extend_from_slice(&row[..2 * RANK * 2]);
            ctx.extend_from_slice(&row[2 * RANK * 2..]);
        }
        let fold_w_bytes: Vec<u8> = fold.w_bf16.iter().flat_map(|w| w.to_le_bytes()).collect();
        let s = Scratch::new(gpu)?;
        Ok(Self {
            schema: DsparkTapSchema {
                taps: TAP_VISITS.to_vec(),
                dim: HIDDEN,
            },
            layers,
            norm: f32w(gpu, "norm.weight")?,
            mask_embed: f32w(gpu, "mask_embed")?,
            markov_norm: f32w(gpu, "markov_head.dflare_norm.weight")?,
            alpha,
            markov_w1: upload_bf16(
                gpu,
                bytes("markov_head.markov_w1.weight")?,
                &[PADDED_VOCAB, RANK],
            )?,
            rnn_w_ctx: upload_bf16(gpu, &ctx, &[3 * RANK, 2 * HIDDEN])?,
            rnn_w_rec: upload_bf16(gpu, &rec, &[3 * RANK, 2 * RANK])?,
            rnn_b: f32w(gpu, "markov_head.rnn_proj.bias")?,
            fold_w: upload_bf16(gpu, &fold_w_bytes, &[VOCAB, RANK])?,
            fold_b: upload_f32(gpu, &fold.b)?,
            head,
            tap_ring: alloc(gpu, TAP_RING * TAP_ROW)?,
            s,
            len: 0,
            position_capacity,
            trace_enabled: false,
            trace: InklingDraftTrace::default(),
            lin: Lin::new(gpu),
        })
    }

    /// `bf16(softmax(fuse_weight))` rows (draft layers 0..3, then Markov).
    pub fn alpha(&self) -> &[[f32; N_TAP]; N_LAYER + 1] {
        &self.alpha
    }

    /// Record `f_rnn` and the Markov states of each window (costs D2H copies).
    pub fn set_trace(&mut self, on: bool) {
        self.trace_enabled = on;
    }

    /// What the last scored window read.
    pub fn trace(&self) -> &InklingDraftTrace {
        &self.trace
    }

    /// `out = Σ_t alpha[t] · src_t` (F32, tap order) for `n` values, where tap
    /// `t`'s values start `t * stride` elements into `src`.
    fn mix(
        gpu: &mut Gpu,
        alpha: &[f32; N_TAP],
        src: &GpuTensor,
        stride: usize,
        n: usize,
        out: &GpuTensor,
        tmp: &GpuTensor,
    ) -> Result<(), String> {
        let out_v = view(out, 0, &[n]);
        let tmp_v = view(tmp, 0, &[n]);
        copy(gpu, out, 0, src, 0, n)?;
        gpu.scale_f32(&out_v, alpha[0]).map_err(hip("mix scale"))?;
        for (t, &a) in alpha.iter().enumerate().skip(1) {
            copy(gpu, tmp, 0, src, t * stride, n)?;
            gpu.scale_f32(&tmp_v, a).map_err(hip("mix scale"))?;
            gpu.add_inplace_f32(&out_v, &tmp_v)
                .map_err(hip("mix add"))?;
        }
        Ok(())
    }

    /// Make room in `layer` for committed rows up to `new_len` plus the
    /// transient block tail.
    fn reserve(
        gpu: &mut Gpu,
        layer: &mut Layer,
        len: usize,
        new_len: usize,
        cap: usize,
    ) -> Result<(), String> {
        if new_len - layer.base + BLOCK <= layer.slots {
            return Ok(());
        }
        if layer.local {
            // Keep the last LOCAL_KEEP rows: a full window before any anchor a
            // one-window truncation can return to. Source and destination do
            // not overlap (`slots >= 2 * LOCAL_KEEP + INGEST_ROWS + BLOCK`).
            let keep = LOCAL_KEEP.min(len);
            let from = len - keep - layer.base;
            for t in [&layer.k_ctx, &layer.v_ctx] {
                copy(gpu, t, 0, t, from * KVD, keep * KVD)?;
            }
            layer.base = len - keep;
            if new_len - layer.base + BLOCK > layer.slots {
                return Err("inkling dspark: local context overflow".into());
            }
            return Ok(());
        }
        let slots = (2 * (layer.slots - BLOCK)).max(new_len).min(cap) + BLOCK;
        if new_len + BLOCK > slots {
            return Err(format!(
                "inkling dspark: {new_len} committed rows exceed the position capacity {cap}"
            ));
        }
        let k = alloc(gpu, slots * KVD)?;
        let v = alloc(gpu, slots * KVD)?;
        copy(gpu, &k, 0, &layer.k_ctx, 0, len * KVD)?;
        copy(gpu, &v, 0, &layer.v_ctx, 0, len * KVD)?;
        let _ = gpu.free_tensor(std::mem::replace(&mut layer.k_ctx, k));
        let _ = gpu.free_tensor(std::mem::replace(&mut layer.v_ctx, v));
        layer.slots = slots;
        Ok(())
    }

    /// Ingest committed rows `[start, start + rows)`, `rows <= INGEST_ROWS`,
    /// from `taps` rows `src_row..`.
    fn ingest_pass(
        &mut self,
        gpu: &mut Gpu,
        taps: &GpuTensor,
        src_row: usize,
        start: usize,
        rows: usize,
    ) -> Result<(), String> {
        let s = &self.s;
        copy(gpu, &s.taps, 0, taps, src_row * TAP_ROW, rows * TAP_ROW)?;
        round(gpu, &s.taps, rows * TAP_ROW)?;
        for r in rows.saturating_sub(TAP_RING)..rows {
            let slot = (start + r) % TAP_RING;
            copy(
                gpu,
                &self.tap_ring,
                slot * TAP_ROW,
                &s.taps,
                r * TAP_ROW,
                TAP_ROW,
            )?;
        }
        // Tap-major copy so each tap of every row is one contiguous span.
        for t in 0..N_TAP {
            gpu.memcpy_dtod_2d_at_ordered_async(
                &s.tap_major.buf,
                t * rows * HIDDEN * 4,
                HIDDEN * 4,
                &s.taps.buf,
                t * HIDDEN * 4,
                TAP_ROW * 4,
                HIDDEN * 4,
                rows,
            )
            .map_err(hip("tap-major copy"))?;
        }
        let pos: Vec<i32> = (start..start + rows).map(|p| p as i32).collect();
        upload_i32(gpu, &s.ctx_pos, &pos)?;
        let n = rows * HIDDEN;
        let mixed = view(&s.mixed, 0, &[rows, HIDDEN]);
        let k = view(&s.ctx_k, 0, &[rows, KVD]);
        let v = view(&s.ctx_v, 0, &[rows, KVD]);
        let cap = self.position_capacity;
        for (l, layer) in self.layers.iter_mut().enumerate() {
            Self::reserve(gpu, layer, start, start + rows, cap)?;
            Self::mix(
                gpu,
                &self.alpha[l],
                &s.tap_major,
                rows * HIDDEN,
                n,
                &s.mixed,
                &s.mix_tmp,
            )?;
            gpu.rmsnorm_batched(&mixed, &layer.dflare_norm, &mixed, rows, HIDDEN, EPS)
                .map_err(hip("dflare_norm"))?;
            round(gpu, &mixed, n)?;
            self.lin
                .run(gpu, &layer.wk_ctx, &mixed, &k, KVD, HIDDEN, rows)
                .map_err(hip("k_proj_ctx"))?;
            self.lin
                .run(gpu, &layer.wv_ctx, &mixed, &v, KVD, HIDDEN, rows)
                .map_err(hip("v_proj_ctx"))?;
            round(gpu, &k, rows * KVD)?;
            round(gpu, &v, rows * KVD)?;
            gpu.rmsnorm_batched(&k, &layer.k_norm_ctx, &k, rows * N_KV, HEAD_DIM, EPS)
                .map_err(hip("k_norm_ctx"))?;
            round(gpu, &k, rows * KVD)?;
            gpu.rope_batched_f32(&k, &k, &s.ctx_pos, 0, N_KV, HEAD_DIM, ROPE_THETA, rows)
                .map_err(hip("ctx rope"))?;
            round(gpu, &k, rows * KVD)?;
            let slot = start - layer.base;
            copy(gpu, &layer.k_ctx, slot * KVD, &k, 0, rows * KVD)?;
            copy(gpu, &layer.v_ctx, slot * KVD, &v, 0, rows * KVD)?;
        }
        self.len = start + rows;
        Ok(())
    }

    /// One draft layer over the block in `s.h` (normalized input) and
    /// `s.residual`, leaving the next normalized input in `s.h`.
    fn block_layer(&mut self, gpu: &mut Gpu, l: usize, position: usize) -> Result<(), String> {
        let s = &self.s;
        let layer = &self.layers[l];
        let h = view(&s.h, 0, &[BLOCK, HIDDEN]);
        let q = view(&s.q, 0, &[BLOCK, QD]);
        let k = view(&s.k, 0, &[BLOCK, KVD]);
        let v = view(&s.v, 0, &[BLOCK, KVD]);
        self.lin
            .run(gpu, &layer.wq, &h, &q, QD, HIDDEN, BLOCK)
            .map_err(hip("q_proj"))?;
        self.lin
            .run(gpu, &layer.wk, &h, &k, KVD, HIDDEN, BLOCK)
            .map_err(hip("k_proj"))?;
        self.lin
            .run(gpu, &layer.wv, &h, &v, KVD, HIDDEN, BLOCK)
            .map_err(hip("v_proj"))?;
        round(gpu, &q, BLOCK * QD)?;
        round(gpu, &k, BLOCK * KVD)?;
        round(gpu, &v, BLOCK * KVD)?;
        gpu.rmsnorm_batched(&q, &layer.q_norm, &q, BLOCK * N_HEAD, HEAD_DIM, EPS)
            .map_err(hip("q_norm"))?;
        gpu.rmsnorm_batched(&k, &layer.k_norm, &k, BLOCK * N_KV, HEAD_DIM, EPS)
            .map_err(hip("k_norm"))?;
        round(gpu, &q, BLOCK * QD)?;
        round(gpu, &k, BLOCK * KVD)?;
        gpu.rope_batched_f32(
            &q, &k, &s.blk_pos, N_HEAD, N_KV, HEAD_DIM, ROPE_THETA, BLOCK,
        )
        .map_err(hip("block rope"))?;
        round(gpu, &q, BLOCK * QD)?;
        round(gpu, &k, BLOCK * KVD)?;
        // The block's K/V sit in the transient slots right after the committed
        // rows, so the keys `[lo, P + BLOCK)` are one contiguous span.
        let tail = position - layer.base;
        copy(gpu, &layer.k_ctx, tail * KVD, &k, 0, BLOCK * KVD)?;
        copy(gpu, &layer.v_ctx, tail * KVD, &v, 0, BLOCK * KVD)?;
        let lo = if layer.local {
            position.saturating_sub(LOCAL_WINDOW)
        } else {
            0
        };
        if lo < layer.base {
            return Err(format!(
                "inkling dspark: layer {l} window starts at {lo} below its retained base {}",
                layer.base
            ));
        }
        let nk = position - lo + BLOCK;
        let kk = view(&layer.k_ctx, (lo - layer.base) * KVD, &[nk, KVD]);
        let vv = view(&layer.v_ctx, (lo - layer.base) * KVD, &[nk, KVD]);
        let attn = view(&s.attn, 0, &[BLOCK, QD]);
        gpu.attention_dflash_f32(&q, &kk, &vv, &attn, BLOCK, nk, N_HEAD, N_KV, HEAD_DIM)
            .map_err(hip("block attention"))?;
        round(gpu, &attn, BLOCK * QD)?;
        let n = BLOCK * HIDDEN;
        let o = view(&s.o, 0, &[BLOCK, HIDDEN]);
        self.lin
            .run(gpu, &layer.wo, &attn, &o, HIDDEN, QD, BLOCK)
            .map_err(hip("o_proj"))?;
        round(gpu, &o, n)?;
        // fused_add_rmsnorm: residual += o (stored BF16), norm of the F32 sum.
        let sum = view(&s.sum, 0, &[BLOCK, HIDDEN]);
        let residual = view(&s.residual, 0, &[BLOCK, HIDDEN]);
        gpu.add_f32(&residual, &o, &sum)
            .map_err(hip("attn residual"))?;
        copy(gpu, &residual, 0, &sum, 0, n)?;
        round(gpu, &residual, n)?;
        gpu.rmsnorm_batched(&sum, &layer.post_ln, &h, BLOCK, HIDDEN, EPS)
            .map_err(hip("post_attention_layernorm"))?;
        round(gpu, &h, n)?;
        let g = view(&s.g, 0, &[BLOCK, FFN]);
        let u = view(&s.u, 0, &[BLOCK, FFN]);
        let act = view(&s.act, 0, &[BLOCK, FFN]);
        self.lin
            .run(gpu, &layer.w_gate, &h, &g, FFN, HIDDEN, BLOCK)
            .map_err(hip("gate_proj"))?;
        self.lin
            .run(gpu, &layer.w_up, &h, &u, FFN, HIDDEN, BLOCK)
            .map_err(hip("up_proj"))?;
        round(gpu, &g, BLOCK * FFN)?;
        round(gpu, &u, BLOCK * FFN)?;
        gpu.silu_mul_f32(&g, &u, &act).map_err(hip("silu_mul"))?;
        round(gpu, &act, BLOCK * FFN)?;
        self.lin
            .run(gpu, &layer.w_down, &act, &o, HIDDEN, FFN, BLOCK)
            .map_err(hip("down_proj"))?;
        round(gpu, &o, n)?;
        gpu.add_f32(&residual, &o, &sum)
            .map_err(hip("mlp residual"))?;
        copy(gpu, &residual, 0, &sum, 0, n)?;
        round(gpu, &residual, n)?;
        let next = if l + 1 < N_LAYER {
            &self.layers[l + 1].input_ln
        } else {
            &self.norm
        };
        gpu.rmsnorm_batched(&sum, next, &h, BLOCK, HIDDEN, EPS)
            .map_err(hip("next norm"))?;
        round(gpu, &h, n)
    }
}

/// First maximum; NaN never wins.
fn argmax_first(v: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        let x = if x.is_nan() { f32::NEG_INFINITY } else { x };
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    best as u32
}

impl DsparkCommittedBody for InklingDsparkBody {
    fn tap_schema(&self) -> &DsparkTapSchema {
        &self.schema
    }

    fn block_size(&self) -> usize {
        BLOCK
    }

    fn proposal_capacity(&self) -> usize {
        K_MAX
    }

    fn committed_len(&self) -> usize {
        self.len
    }

    fn position_capacity(&self) -> usize {
        self.position_capacity
    }

    fn ingest_committed(
        &mut self,
        gpu: &mut Gpu,
        taps: &GpuTensor,
        start: usize,
        rows: usize,
    ) -> Result<(), String> {
        if start != self.len {
            return Err(format!(
                "inkling dspark: ingest at {start} but {} rows are committed",
                self.len
            ));
        }
        if taps.numel() < rows * TAP_ROW || taps.dtype != DType::F32 {
            return Err("inkling dspark: tap rows shorter than the schema".into());
        }
        if start + rows + BLOCK > self.position_capacity + BLOCK {
            return Err(format!(
                "inkling dspark: committed rows through {} exceed the position capacity {}",
                start + rows,
                self.position_capacity
            ));
        }
        let mut done = 0;
        while done < rows {
            let n = (rows - done).min(INGEST_ROWS);
            self.ingest_pass(gpu, taps, done, start + done, n)?;
            done += n;
        }
        Ok(())
    }

    fn truncate_committed(&mut self, _gpu: &mut Gpu, len: usize) -> Result<(), String> {
        if len > self.len || self.len - len > BLOCK {
            return Err(format!(
                "inkling dspark: truncate {} committed rows to {len} (at most one window)",
                self.len
            ));
        }
        self.len = len;
        Ok(())
    }

    fn draft_block(
        &mut self,
        gpu: &mut Gpu,
        seed: u32,
        position: usize,
        x_head: &GpuTensor,
    ) -> Result<(), String> {
        if position != self.len {
            return Err(format!(
                "inkling dspark: anchor at {position} but {} rows are committed",
                self.len
            ));
        }
        if position == 0 {
            return Err("inkling dspark: no committed row before the anchor".into());
        }
        if position + BLOCK > self.position_capacity {
            return Err("inkling dspark: block crosses the position capacity".into());
        }
        let n = BLOCK * HIDDEN;
        self.head.anchor_embedding(gpu, seed, &self.s.x)?;
        for j in 1..BLOCK {
            copy(gpu, &self.s.x, j * HIDDEN, &self.mask_embed, 0, HIDDEN)?;
        }
        copy(gpu, &self.s.residual, 0, &self.s.x, 0, n)?;
        let pos: Vec<i32> = (position..position + BLOCK).map(|p| p as i32).collect();
        upload_i32(gpu, &self.s.blk_pos, &pos)?;
        let x = view(&self.s.x, 0, &[BLOCK, HIDDEN]);
        let h = view(&self.s.h, 0, &[BLOCK, HIDDEN]);
        gpu.rmsnorm_batched(&x, &self.layers[0].input_ln, &h, BLOCK, HIDDEN, EPS)
            .map_err(hip("input_layernorm"))?;
        round(gpu, &h, n)?;
        for l in 0..N_LAYER {
            self.block_layer(gpu, l, position)?;
        }
        if x_head.numel() < n {
            return Err("inkling dspark: x_head shorter than the block".into());
        }
        copy(gpu, x_head, 0, &self.s.h, 0, n)
    }

    fn score_block(
        &mut self,
        gpu: &mut Gpu,
        x_head: &GpuTensor,
        seed: u32,
        k: usize,
    ) -> Result<DraftResult, String> {
        if k == 0 || k > K_MAX {
            return Err(format!("inkling dspark: k {k} outside 1..={K_MAX}"));
        }
        if self.len == 0 {
            return Err("inkling dspark: no committed row for f_rnn".into());
        }
        let s = &self.s;
        // f_rnn from the last committed row, P - 1 (never the anchor's row).
        let frnn_row = self.len - 1;
        let ring = view(&self.tap_ring, (frnn_row % TAP_RING) * TAP_ROW, &[TAP_ROW]);
        Self::mix(
            gpu,
            &self.alpha[N_LAYER],
            &ring,
            HIDDEN,
            HIDDEN,
            &s.frnn,
            &s.mix_tmp,
        )?;
        let frnn = view(&s.frnn, 0, &[1, HIDDEN]);
        gpu.rmsnorm_batched(&frnn, &self.markov_norm, &frnn, 1, HIDDEN, EPS)
            .map_err(hip("markov dflare_norm"))?;
        round(gpu, &frnn, HIDDEN)?;
        // Base logits of fill-in rows 1..=k.
        let rows = view(x_head, HIDDEN, &[k, HIDDEN]);
        let base = view(&s.base, 0, &[k, VOCAB]);
        self.head.unembed(gpu, &rows, k, &base)?;
        round(gpu, &base, k * VOCAB)?;
        // Per-row operand: rnn_b + [hidden_j, f_rnn] · W[:, 2R:].
        for j in 0..k {
            copy(
                gpu,
                &s.xctx,
                j * 2 * HIDDEN,
                x_head,
                (j + 1) * HIDDEN,
                HIDDEN,
            )?;
            copy(gpu, &s.xctx, j * 2 * HIDDEN + HIDDEN, &s.frnn, 0, HIDDEN)?;
        }
        let xctx = view(&s.xctx, 0, &[k, 2 * HIDDEN]);
        let ctx_raw = view(&s.ctx_raw, 0, &[k, 3 * RANK]);
        self.lin
            .run(
                gpu,
                &self.rnn_w_ctx,
                &xctx,
                &ctx_raw,
                3 * RANK,
                2 * HIDDEN,
                k,
            )
            .map_err(hip("rnn_proj ctx"))?;
        for j in 0..k {
            let row = view(&s.ctx_raw, j * 3 * RANK, &[3 * RANK]);
            gpu.add_inplace_f32(&row, &self.rnn_b)
                .map_err(hip("rnn_proj bias"))?;
        }
        let state = view(&s.sw, 0, &[RANK]);
        let w1 = view(&s.sw, RANK, &[RANK]);
        gpu.fill_f32(&state, 0.0).map_err(hip("zero state"))?;
        let gate = view(&s.raw, 0, &[RANK]);
        let cand = view(&s.raw, RANK, &[RANK]);
        let co = view(&s.raw, RANK, &[2 * RANK]);
        let latent = view(&s.raw, 2 * RANK, &[1, RANK]);
        let bias = view(&s.bias, 0, &[VOCAB]);
        let logits = view(&s.logits, 0, &[VOCAB]);
        let mut trace = InklingDraftTrace {
            frnn_row: Some(frnn_row),
            ..Default::default()
        };
        if self.trace_enabled {
            trace.frnn = gpu
                .download_f32(&view(&s.frnn, 0, &[HIDDEN]))
                .map_err(hip("trace f_rnn"))?;
        }
        let mut prev = seed;
        let mut tokens = Vec::with_capacity(k);
        let mut host = Vec::with_capacity(VOCAB);
        for step in 0..k {
            if prev as usize >= PADDED_VOCAB {
                return Err(format!(
                    "inkling dspark: previous id {prev} outside the vocabulary"
                ));
            }
            upload_i32(gpu, &s.prev, &[prev as i32])?;
            gpu.embedding_lookup_bf16_batched(&self.markov_w1, &w1, &s.prev, 1, RANK)
                .map_err(hip("markov_w1"))?;
            let sw = view(&s.sw, 0, &[1, 2 * RANK]);
            let rec = view(&s.rec, 0, &[1, 3 * RANK]);
            self.lin
                .run(gpu, &self.rnn_w_rec, &sw, &rec, 3 * RANK, 2 * RANK, 1)
                .map_err(hip("rnn_proj rec"))?;
            let ctx_row = view(&s.ctx_raw, step * 3 * RANK, &[3 * RANK]);
            let raw = view(&s.raw, 0, &[3 * RANK]);
            gpu.add_f32(&ctx_row, &rec, &raw).map_err(hip("rnn raw"))?;
            round(gpu, &raw, 3 * RANK)?;
            // markov_gate: gate = σ, candidate and output = tanh, BF16 between.
            gpu.sigmoid_f32(&gate).map_err(hip("gate sigmoid"))?;
            gpu.logit_softcap_f32(&co, 2 * RANK, 1.0)
                .map_err(hip("tanh"))?;
            round(gpu, &raw, 3 * RANK)?;
            gpu.mul_f32(&gate, &state, &s.keep).map_err(hip("keep"))?;
            round(gpu, &s.keep, RANK)?;
            copy(gpu, &s.omg, 0, &s.raw, 0, RANK)?;
            gpu.scale_f32(&s.omg, -1.0).map_err(hip("1 - gate"))?;
            gpu.add_f32(&s.ones, &s.omg, &s.omg)
                .map_err(hip("1 - gate"))?;
            round(gpu, &s.omg, RANK)?;
            gpu.mul_f32(&s.omg, &cand, &s.upd).map_err(hip("update"))?;
            round(gpu, &s.upd, RANK)?;
            gpu.add_f32(&s.keep, &s.upd, &state).map_err(hip("state"))?;
            round(gpu, &state, RANK)?;
            // Folded hidden_out correction, then the target softcap.
            self.lin
                .run(gpu, &self.fold_w, &latent, &bias, VOCAB, RANK, 1)
                .map_err(hip("fold"))?;
            gpu.add_inplace_f32(&bias, &self.fold_b)
                .map_err(hip("fold bias"))?;
            round(gpu, &bias, VOCAB)?;
            let base_row = view(&s.base, step * VOCAB, &[VOCAB]);
            gpu.add_f32(&base_row, &bias, &logits)
                .map_err(hip("logits"))?;
            round(gpu, &logits, VOCAB)?;
            gpu.logit_softcap_f32(&logits, VOCAB, SOFTCAP)
                .map_err(hip("softcap"))?;
            round(gpu, &logits, VOCAB)?;
            gpu.download_f32_into(&logits, &mut host)
                .map_err(hip("logits d2h"))?;
            if self.trace_enabled {
                trace
                    .states
                    .push(gpu.download_f32(&state).map_err(hip("trace state"))?);
                trace.logits.push(host.clone());
            }
            prev = argmax_first(&host);
            tokens.push(prev);
        }
        self.trace = trace;
        Ok(DraftResult {
            tokens,
            logits: Vec::new(),
            confidence: Vec::new(),
        })
    }

    fn reset(&mut self, _gpu: &mut Gpu) -> Result<(), String> {
        self.len = 0;
        for layer in &mut self.layers {
            layer.base = 0;
        }
        self.trace = InklingDraftTrace::default();
        Ok(())
    }

    fn free(self: Box<Self>, gpu: &mut Gpu) {
        let me = *self;
        me.head.free(gpu);
        for layer in me.layers {
            for t in layer.tensors() {
                let _ = gpu.free_tensor(t);
            }
        }
        for t in me.s.tensors() {
            let _ = gpu.free_tensor(t);
        }
        for t in [
            me.norm,
            me.mask_embed,
            me.markov_norm,
            me.markov_w1,
            me.rnn_w_ctx,
            me.rnn_w_rec,
            me.rnn_b,
            me.fold_w,
            me.fold_b,
            me.tap_ring,
        ] {
            let _ = gpu.free_tensor(t);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_rounding_matches_the_reference() {
        assert_eq!(bf16_bits(1.0), 0x3F80);
        // Ties to even: 1 + 2^-8 sits halfway between 1 and 1 + 2^-7.
        assert_eq!(bf16_bits(1.0 + 1.0 / 256.0), 0x3F80);
        assert_eq!(bf16_bits(1.0 + 3.0 / 256.0), 0x3F82);
        assert_eq!(bf16_to_f32(bf16_bits(-2.5)), -2.5);
        // F64 rounds once: just above a tie rounds up even when F32 lands on it.
        let tie = 1.0f64 + 1.0 / 256.0;
        assert_eq!(bf16_from_f64(tie + 1e-12), 0x3F81);
        assert_eq!(bf16_from_f64(tie), 0x3F80);
    }

    #[test]
    fn taps_are_the_inputs_to_visits_5_23_41_and_the_final_norm() {
        assert_eq!(TAP_VISITS, [5, 23, 41, 48]);
        assert_eq!(TAP_ROW, 4 * 2304);
    }

    #[test]
    fn fold_accumulates_in_f64_and_rounds_once() {
        let mut hw = vec![0.0f32; HIDDEN * RANK];
        let mut hb = vec![0.0f32; HIDDEN];
        hw[0] = 1.0; // w[v][0] = U[v][0]
        hw[RANK + 1] = 0.5; // w[v][1] = U[v][1] / 2
        hb[0] = 2.0;
        let row = |v: usize, out: &mut [f32]| {
            out.fill(0.0);
            out[0] = v as f32;
            out[1] = 3.0;
        };
        let f = InklingFold::compute(&row, &hw, &hb).unwrap();
        assert_eq!(f.w_bf16.len(), VOCAB * RANK);
        assert_eq!(bf16_to_f32(f.w_bf16[7 * RANK]), 7.0);
        assert_eq!(bf16_to_f32(f.w_bf16[7 * RANK + 1]), 1.5);
        assert_eq!(f.b[7], 14.0);
        // 1001 is not BF16-representable: one RNE rounding.
        assert_eq!(f.b[1001] as f64, bf16_to_f32(bf16_from_f64(2002.0)) as f64);
    }

    #[test]
    fn argmax_takes_the_first_maximum_and_never_nan() {
        assert_eq!(argmax_first(&[1.0, 3.0, 3.0, 2.0]), 1);
        assert_eq!(argmax_first(&[f32::NAN, 0.5, 0.5]), 1);
    }
}
