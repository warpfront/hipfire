// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU-resident FLUX VAE decoder (latent -> pixels), the GPU companion of the
//! CPU [`crate::vae`] reference. It uploads the host [`VaeDecoderWeights`]
//! once (`GpuVaeDecoderWeights::from_host`) and then runs the exact same
//! stage walk — `conv_in`, mid resnet/attn/resnet, processing-ordered up
//! blocks, `norm_out`, SiLU, `conv_out` — through the f32 `vae_*` kernels in
//! `rdna_compute`, keeping the CPU summation structure so the parity gate
//! against the CPU decode and ComfyUI stays a math check, not a rounding
//! audit.
//!
//! The CPU decoder is single-threaded naive convolutions and takes minutes
//! for a 1024x1024 image; this path brings the decode to GPU speed. It is
//! correctness-first (plain FMA kernels, no WMMA) exactly like the CPU
//! reference it mirrors.

use crate::flux::Tensor;
use crate::vae::{MidAttn, VaeConfig, VaeDecoderWeights, VaeEncoderWeights, VaeResnet};
use rdna_compute::{DType, Gpu, GpuTensor};

/// GPU-resident resnet weights (f32). Shapes match the host [`VaeResnet`]:
/// convs `[c_out][c_in*9]`, norms `[c]`, optional 1x1 shortcut `[out][in]`.
pub struct GpuVaeResnet {
    pub in_ch: usize,
    pub out_ch: usize,
    pub norm1_w: GpuTensor,
    pub norm1_b: GpuTensor,
    pub conv1_w: GpuTensor,
    pub conv1_b: GpuTensor,
    pub norm2_w: GpuTensor,
    pub norm2_b: GpuTensor,
    pub conv2_w: GpuTensor,
    pub conv2_b: GpuTensor,
    pub shortcut_w: Option<GpuTensor>,
    pub shortcut_b: Option<GpuTensor>,
}

/// GPU-resident up block: resnets in order, then the optional upsample conv.
pub struct GpuUpBlock {
    pub channels: usize,
    pub resnets: Vec<GpuVaeResnet>,
    pub upsample_w: Option<GpuTensor>,
    pub upsample_b: Option<GpuTensor>,
}

/// GPU-resident encoder down block: resnets in order, then the optional
/// stride-2 downsample conv (absent only on the last block). Mirrors the host
/// [`crate::vae::DownBlock`].
pub struct GpuDownBlock {
    pub channels: usize,
    pub resnets: Vec<GpuVaeResnet>,
    pub downsample_w: Option<GpuTensor>,
    pub downsample_b: Option<GpuTensor>,
}

/// GPU-resident mid attention (1x1 q/k/v/proj as `[c][c]` linears).
pub struct GpuMidAttn {
    pub group_norm_w: GpuTensor,
    pub group_norm_b: GpuTensor,
    pub q_w: GpuTensor,
    pub q_b: GpuTensor,
    pub k_w: GpuTensor,
    pub k_b: GpuTensor,
    pub v_w: GpuTensor,
    pub v_b: GpuTensor,
    pub out_w: GpuTensor,
    pub out_b: GpuTensor,
}

/// GPU-resident VAE decoder weights, mirroring [`VaeDecoderWeights`].
///
/// Uploaded once per process by [`crate::pipeline::FluxPipeBundle::ensure_gpu`]
/// and kept resident; the decode used to re-upload all ~200 MB per generation.
pub struct GpuVaeDecoderWeights {
    pub config: VaeConfig,
    /// True when every conv weight is `DType::F16` — the WMMA GEMM's operand
    /// dtype, so the decode reads them straight out of residency instead of
    /// casting a fresh f16 copy per conv per generation. False means the
    /// weights are f32 and the decode must take the direct-conv route
    /// (`HIPFIRE_VAE_CONV=direct`, or a K the GEMM cannot take).
    pub conv_w_f16: bool,
    pub conv_in_w: GpuTensor,
    pub conv_in_b: GpuTensor,
    pub mid_resnet: Vec<GpuVaeResnet>,
    pub mid_attn: Option<GpuMidAttn>,
    pub up_blocks: Vec<GpuUpBlock>,
    pub conv_norm_out_w: GpuTensor,
    pub conv_norm_out_b: GpuTensor,
    pub conv_out_w: GpuTensor,
    pub conv_out_b: GpuTensor,
    /// FLUX.2 Klein's `post_quant_conv` (1x1, latent->latent), applied to the
    /// latent before `conv_in`. `None` for FLUX.1, whose decode then issues
    /// exactly the launch sequence it always did.
    pub post_quant_conv: Option<(GpuTensor, GpuTensor)>,
}

/// GPU-resident VAE encoder weights, mirroring [`VaeEncoderWeights`].
///
/// Unlike the decoder these are always f32: the encoder takes the direct
/// per-thread conv route ([`GpuVaeDecoderWeights::conv_w_f16`]'s `false`
/// branch). It runs once per reference image — the decoder's im2col + WMMA
/// GEMM route exists because a 1024x1024 decode runs it dozens of times per
/// generation, which is not the shape of this workload.
pub struct GpuVaeEncoderWeights {
    pub config: VaeConfig,
    pub conv_in_w: GpuTensor,
    pub conv_in_b: GpuTensor,
    /// Indexed 0..nb, processed in order (spatial /2 at every downsampler).
    pub down_blocks: Vec<GpuDownBlock>,
    pub mid_resnet: Vec<GpuVaeResnet>,
    pub mid_attn: Option<GpuMidAttn>,
    pub conv_norm_out_w: GpuTensor,
    pub conv_norm_out_b: GpuTensor,
    /// `[2*latent][block_out[nb-1]*9]` — the encoder emits mean AND logvar.
    pub conv_out_w: GpuTensor,
    pub conv_out_b: GpuTensor,
    /// Top-level `quant_conv` (1x1, `2*latent -> 2*latent`).
    pub quant_conv: Option<(GpuTensor, GpuTensor)>,
}

/// Every device tensor [`GpuVaeDecoderWeights::from_host`] has uploaded so far.
///
/// **Invariant: from the moment a tensor is allocated on the device until the
/// finished `GpuVaeDecoderWeights` is returned, the ledger owns it.** Nothing
/// on the upload path holds a `GpuTensor` directly; uploads return a SLOT
/// index, and the result structure is assembled in one infallible pass at the
/// end by [`take`](Self::take)-ing those slots.
///
/// Why it exists: `from_host` performs ~150 allocations, `GpuTensor` has no
/// `Drop`, and a `?` partway through used to strand every tensor uploaded so
/// far for the life of the process. `ensure_gpu` propagates that error, so the
/// next `img_generate` retried and leaked the whole partial decoder again —
/// on a device that was already out of memory. Mirrors the `free_partial`
/// pattern of `flux_gpu::GpuFluxWeights::from_stream`.
#[derive(Default)]
struct UploadLedger {
    slots: Vec<Option<GpuTensor>>,
}

impl UploadLedger {
    /// Take ownership of one freshly-uploaded tensor; returns its slot.
    fn record(&mut self, t: GpuTensor) -> usize {
        self.slots.push(Some(t));
        self.slots.len() - 1
    }

    /// Borrow a recorded tensor (used between a staged upload and its cast).
    fn get(&self, slot: usize) -> &GpuTensor {
        self.slots[slot]
            .as_ref()
            .unwrap_or_else(|| panic!("vae gpu: ledger slot {slot} is empty"))
    }

    /// Move a recorded tensor out into the finished structure.
    fn take(&mut self, slot: usize) -> GpuTensor {
        self.slots[slot]
            .take()
            .unwrap_or_else(|| panic!("vae gpu: ledger slot {slot} taken twice"))
    }

    /// Free one recorded tensor early (the f32 staging copy of an f16 conv).
    fn free_slot(&mut self, gpu: &mut Gpu, slot: usize, what: &str) -> Result<(), String> {
        let t = self.take(slot);
        gpu.free_tensor(t)
            .map_err(|e| format!("vae gpu: free f32 staging {what}: {e:?}"))
    }

    /// How many tensors the ledger still owns — i.e. exactly what
    /// [`free_partial`](Self::free_partial) would return to the pool.
    #[cfg(test)]
    fn outstanding(&self) -> usize {
        self.slots.iter().filter(|s| s.is_some()).count()
    }

    /// Best-effort: return every tensor still held to the pool, and report how
    /// many were freed.
    ///
    /// Deliberately does NOT panic on a failed free — this runs while
    /// unwinding a DIFFERENT error, and replacing "device out of memory
    /// uploading `up[3].resnet[1].conv2.weight`" with a panic from the cleanup
    /// would destroy the only useful diagnostic. Same reasoning as
    /// `flux_gpu::free_partial`.
    fn free_partial(self, gpu: &mut Gpu) -> usize {
        let mut freed = 0usize;
        for t in self.slots.into_iter().flatten() {
            if gpu.free_tensor(t).is_ok() {
                freed += 1;
            }
        }
        freed
    }
}

fn up(gpu: &mut Gpu, led: &mut UploadLedger, t: &Tensor, what: &str) -> Result<usize, String> {
    if t.data.len() != t.rows * t.cols {
        return Err(format!(
            "vae gpu: {what}: host tensor has {} elems, shape says {}",
            t.data.len(),
            t.rows * t.cols
        ));
    }
    let g = gpu
        .upload_f32(&t.data, &[t.rows, t.cols])
        .map_err(|e| format!("vae gpu: upload {what}: {e:?}"))?;
    Ok(led.record(g))
}

/// Upload one conv weight in the dtype the decode's conv route reads.
///
/// `f16` uploads an f32 staging copy, casts it once, and returns the staging
/// buffer to the pool — the same shape as `flux_gpu::upload_flux_tensor`. The
/// decode then hands the resident tensor straight to the WMMA GEMM. The
/// previous code kept the weights f32 and cast a fresh f16 copy inside every
/// conv of every decode (the `wcast` profile line).
///
/// Both the staging copy and the f16 result are recorded in `led` before the
/// cast, so a failure between the two allocations still frees both.
fn up_conv(
    gpu: &mut Gpu,
    led: &mut UploadLedger,
    t: &Tensor,
    what: &str,
    f16: bool,
) -> Result<usize, String> {
    let staged = up(gpu, led, t, what)?;
    if !f16 {
        return Ok(staged);
    }
    let g = gpu
        .alloc_tensor(&[t.rows, t.cols], DType::F16)
        .map_err(|e| format!("vae gpu: alloc f16 {what}: {e:?}"))?;
    let out = led.record(g);
    gpu.cast_f32_to_f16(led.get(staged), led.get(out))
        .map_err(|e| format!("vae gpu: cast f16 {what}: {e:?}"))?;
    led.free_slot(gpu, staged, what)?;
    Ok(out)
}

/// Slot indices into an [`UploadLedger`], mirroring [`GpuVaeResnet`].
struct SlotResnet {
    in_ch: usize,
    out_ch: usize,
    norm1_w: usize,
    norm1_b: usize,
    conv1_w: usize,
    conv1_b: usize,
    norm2_w: usize,
    norm2_b: usize,
    conv2_w: usize,
    conv2_b: usize,
    shortcut_w: Option<usize>,
    shortcut_b: Option<usize>,
}

impl SlotResnet {
    fn take(self, led: &mut UploadLedger) -> GpuVaeResnet {
        GpuVaeResnet {
            in_ch: self.in_ch,
            out_ch: self.out_ch,
            norm1_w: led.take(self.norm1_w),
            norm1_b: led.take(self.norm1_b),
            conv1_w: led.take(self.conv1_w),
            conv1_b: led.take(self.conv1_b),
            norm2_w: led.take(self.norm2_w),
            norm2_b: led.take(self.norm2_b),
            conv2_w: led.take(self.conv2_w),
            conv2_b: led.take(self.conv2_b),
            shortcut_w: self.shortcut_w.map(|s| led.take(s)),
            shortcut_b: self.shortcut_b.map(|s| led.take(s)),
        }
    }
}

/// Upload one resnet's weights, shared by the decoder and the encoder (the
/// two differ only in which blocks hold the resnets, never in the block).
fn resnet_from_host(
    gpu: &mut Gpu,
    led: &mut UploadLedger,
    r: &VaeResnet,
    what: &str,
    f16: bool,
) -> Result<SlotResnet, String> {
    let (sw, sb) = match (&r.nin_shortcut_w, &r.nin_shortcut_b) {
        (Some(w), Some(b)) => (
            Some(up_conv(
                gpu,
                led,
                w,
                &format!("{what}.shortcut.weight"),
                f16,
            )?),
            Some(up(gpu, led, b, &format!("{what}.shortcut.bias"))?),
        ),
        (None, None) => (None, None),
        _ => return Err(format!("vae gpu: {what}: half-present shortcut")),
    };
    Ok(SlotResnet {
        in_ch: r.in_ch,
        out_ch: r.out_ch,
        norm1_w: up(gpu, led, &r.norm1_w, &format!("{what}.norm1.weight"))?,
        norm1_b: up(gpu, led, &r.norm1_b, &format!("{what}.norm1.bias"))?,
        conv1_w: up_conv(gpu, led, &r.conv1_w, &format!("{what}.conv1.weight"), f16)?,
        conv1_b: up(gpu, led, &r.conv1_b, &format!("{what}.conv1.bias"))?,
        norm2_w: up(gpu, led, &r.norm2_w, &format!("{what}.norm2.weight"))?,
        norm2_b: up(gpu, led, &r.norm2_b, &format!("{what}.norm2.bias"))?,
        conv2_w: up_conv(gpu, led, &r.conv2_w, &format!("{what}.conv2.weight"), f16)?,
        conv2_b: up(gpu, led, &r.conv2_b, &format!("{what}.conv2.bias"))?,
        shortcut_w: sw,
        shortcut_b: sb,
    })
}

/// Slot indices into an [`UploadLedger`], mirroring [`GpuUpBlock`].
struct SlotUpBlock {
    channels: usize,
    resnets: Vec<SlotResnet>,
    upsample_w: Option<usize>,
    upsample_b: Option<usize>,
}

impl SlotUpBlock {
    fn take(self, led: &mut UploadLedger) -> GpuUpBlock {
        GpuUpBlock {
            channels: self.channels,
            resnets: self.resnets.into_iter().map(|r| r.take(led)).collect(),
            upsample_w: self.upsample_w.map(|s| led.take(s)),
            upsample_b: self.upsample_b.map(|s| led.take(s)),
        }
    }
}

/// Slot indices into an [`UploadLedger`], mirroring [`GpuDownBlock`].
struct SlotDownBlock {
    channels: usize,
    resnets: Vec<SlotResnet>,
    downsample_w: Option<usize>,
    downsample_b: Option<usize>,
}

impl SlotDownBlock {
    fn take(self, led: &mut UploadLedger) -> GpuDownBlock {
        GpuDownBlock {
            channels: self.channels,
            resnets: self.resnets.into_iter().map(|r| r.take(led)).collect(),
            downsample_w: self.downsample_w.map(|s| led.take(s)),
            downsample_b: self.downsample_b.map(|s| led.take(s)),
        }
    }
}

/// Slot indices into an [`UploadLedger`], mirroring [`GpuMidAttn`].
struct SlotMidAttn {
    group_norm_w: usize,
    group_norm_b: usize,
    q_w: usize,
    q_b: usize,
    k_w: usize,
    k_b: usize,
    v_w: usize,
    v_b: usize,
    out_w: usize,
    out_b: usize,
}

impl SlotMidAttn {
    fn take(self, led: &mut UploadLedger) -> GpuMidAttn {
        GpuMidAttn {
            group_norm_w: led.take(self.group_norm_w),
            group_norm_b: led.take(self.group_norm_b),
            q_w: led.take(self.q_w),
            q_b: led.take(self.q_b),
            k_w: led.take(self.k_w),
            k_b: led.take(self.k_b),
            v_w: led.take(self.v_w),
            v_b: led.take(self.v_b),
            out_w: led.take(self.out_w),
            out_b: led.take(self.out_b),
        }
    }
}

/// The whole decoder as ledger slots — the "partial structure" the upload
/// builds, converted to the real one only once every upload has succeeded.
struct SlotDecoder {
    conv_in_w: usize,
    conv_in_b: usize,
    mid_resnet: Vec<SlotResnet>,
    mid_attn: Option<SlotMidAttn>,
    up_blocks: Vec<SlotUpBlock>,
    conv_norm_out_w: usize,
    conv_norm_out_b: usize,
    conv_out_w: usize,
    conv_out_b: usize,
    post_quant_conv: Option<(usize, usize)>,
}

/// The whole encoder as ledger slots — same "assemble only once every upload
/// succeeded" discipline as [`SlotDecoder`].
struct SlotEncoder {
    conv_in_w: usize,
    conv_in_b: usize,
    down_blocks: Vec<SlotDownBlock>,
    mid_resnet: Vec<SlotResnet>,
    mid_attn: Option<SlotMidAttn>,
    conv_norm_out_w: usize,
    conv_norm_out_b: usize,
    conv_out_w: usize,
    conv_out_b: usize,
    quant_conv: Option<(usize, usize)>,
}

/// Can every conv in this decoder take the WMMA GEMM route?
///
/// The GEMM needs `K % 16 == 0`, where K is the conv weight's column count
/// (`c_in*9` for a 3x3, `c_in` for a 1x1) — exactly what `Run::conv3x3` /
/// `Run::conv1x1` check per call. Deciding it up front lets the weights be
/// uploaded in the dtype the chosen route reads, and makes the fallback a
/// whole-decoder property instead of a per-conv surprise on a dtype that no
/// longer matches.
fn all_convs_gemm_ready(host: &VaeDecoderWeights) -> bool {
    let ok = |t: &Tensor| t.cols % 16 == 0;
    let resnet_ok = |r: &VaeResnet| {
        ok(&r.conv1_w) && ok(&r.conv2_w) && r.nin_shortcut_w.as_ref().map_or(true, ok)
    };
    ok(&host.conv_in_w)
        && ok(&host.conv_out_w)
        // FLUX.2's post_quant_conv is a 1x1 latent->latent, so its K is the
        // latent width (32 for Klein). It runs through `Run::conv1x1`, which
        // rejects a ragged c_in on the GEMM route just like the others.
        && host
            .post_quant_conv
            .as_ref()
            .map_or(true, |(w, _)| ok(w))
        && host.mid_resnet.iter().all(resnet_ok)
        && host.mid_attn.as_ref().map_or(true, |a| {
            ok(&a.q_w) && ok(&a.k_w) && ok(&a.v_w) && ok(&a.out_w)
        })
        && host.up_blocks.iter().all(|b| {
            b.resnets.iter().all(resnet_ok) && b.upsample_w.as_ref().map_or(true, ok)
        })
}

impl GpuVaeDecoderWeights {
    /// Upload every decoder weight of a loaded host [`VaeDecoderWeights`].
    /// The host copy stays untouched and remains the source of truth.
    ///
    /// Conv weights land as f16 (the GEMM operand dtype) unless the direct
    /// route is pinned by `HIPFIRE_VAE_CONV=direct` or some conv's K is not a
    /// multiple of 16; norms and biases are read as f32 by their kernels and
    /// stay f32.
    ///
    /// **Failure frees what it uploaded.** ~150 device allocations happen here
    /// and `GpuTensor` has no `Drop`; every one of them is owned by an
    /// [`UploadLedger`] until the whole decoder is assembled, so an error at
    /// allocation 90 returns the first 89 to the pool and reports
    /// `… [freed N partially-uploaded tensors]`. Without that, `ensure_gpu`
    /// propagated the error and the caller's next `img_generate` retried
    /// against a device that the previous attempt had already filled.
    pub fn from_host(gpu: &mut Gpu, host: &VaeDecoderWeights) -> Result<Self, String> {
        let gemm =
            hipfire_config::developer_var("HIPFIRE_VAE_CONV").map_or(true, |v| v != "direct");
        let f16 = gemm && all_convs_gemm_ready(host);
        let mut led = UploadLedger::default();
        match Self::upload_into(gpu, &mut led, host, f16) {
            Ok(slots) => Ok(GpuVaeDecoderWeights {
                config: host.config.clone(),
                conv_w_f16: f16,
                conv_in_w: led.take(slots.conv_in_w),
                conv_in_b: led.take(slots.conv_in_b),
                mid_resnet: slots
                    .mid_resnet
                    .into_iter()
                    .map(|r| r.take(&mut led))
                    .collect(),
                mid_attn: slots.mid_attn.map(|a| a.take(&mut led)),
                up_blocks: slots
                    .up_blocks
                    .into_iter()
                    .map(|b| b.take(&mut led))
                    .collect(),
                conv_norm_out_w: led.take(slots.conv_norm_out_w),
                conv_norm_out_b: led.take(slots.conv_norm_out_b),
                conv_out_w: led.take(slots.conv_out_w),
                conv_out_b: led.take(slots.conv_out_b),
                post_quant_conv: slots
                    .post_quant_conv
                    .map(|(w, b)| (led.take(w), led.take(b))),
            }),
            Err(e) => {
                let freed = led.free_partial(gpu);
                Err(format!("{e} [freed {freed} partially-uploaded tensors]"))
            }
        }
    }

    /// The upload loop, split out so `from_host` owns the ledger on the error
    /// path. Uses `?` freely; every early return lands in the cleanup arm.
    fn upload_into(
        gpu: &mut Gpu,
        led: &mut UploadLedger,
        host: &VaeDecoderWeights,
        f16: bool,
    ) -> Result<SlotDecoder, String> {
        let resnet = |gpu: &mut Gpu, led: &mut UploadLedger, r: &VaeResnet, what: &str| {
            resnet_from_host(gpu, led, r, what, f16)
        };
        let mid_attn = match &host.mid_attn {
            None => None,
            Some(a) => Some(mid_attn_from_host(gpu, led, a, f16)?),
        };
        let mut up_blocks = Vec::with_capacity(host.up_blocks.len());
        for (i, b) in host.up_blocks.iter().enumerate() {
            let mut resnets = Vec::with_capacity(b.resnets.len());
            for (j, r) in b.resnets.iter().enumerate() {
                resnets.push(resnet(gpu, led, r, &format!("up[{i}].resnet[{j}]"))?);
            }
            let (uw, ub) = match (&b.upsample_w, &b.upsample_b) {
                (Some(w), Some(bb)) => (
                    Some(up_conv(
                        gpu,
                        led,
                        w,
                        &format!("up[{i}].upsample.weight"),
                        f16,
                    )?),
                    Some(up(gpu, led, bb, &format!("up[{i}].upsample.bias"))?),
                ),
                (None, None) => (None, None),
                _ => return Err(format!("vae gpu: up[{i}]: half-present upsample")),
            };
            up_blocks.push(SlotUpBlock {
                channels: b.channels,
                resnets,
                upsample_w: uw,
                upsample_b: ub,
            });
        }
        let mut mid_resnet = Vec::with_capacity(host.mid_resnet.len());
        for (i, r) in host.mid_resnet.iter().enumerate() {
            mid_resnet.push(resnet(gpu, led, r, &format!("mid_resnet[{i}]"))?);
        }
        let post_quant_conv = match &host.post_quant_conv {
            None => None,
            Some((w, b)) => Some((
                up_conv(gpu, led, w, "post_quant_conv.weight", f16)?,
                up(gpu, led, b, "post_quant_conv.bias")?,
            )),
        };
        Ok(SlotDecoder {
            conv_in_w: up_conv(gpu, led, &host.conv_in_w, "conv_in.weight", f16)?,
            conv_in_b: up(gpu, led, &host.conv_in_b, "conv_in.bias")?,
            mid_resnet,
            mid_attn,
            up_blocks,
            conv_norm_out_w: up(gpu, led, &host.conv_norm_out_w, "norm_out.weight")?,
            conv_norm_out_b: up(gpu, led, &host.conv_norm_out_b, "norm_out.bias")?,
            conv_out_w: up_conv(gpu, led, &host.conv_out_w, "conv_out.weight", f16)?,
            conv_out_b: up(gpu, led, &host.conv_out_b, "conv_out.bias")?,
            post_quant_conv,
        })
    }

    /// Return every GPU buffer to the pool. Exhaustive by construction.
    pub fn free_gpu(self, gpu: &mut Gpu) -> usize {
        let mut freed = 0usize;
        free_t(gpu, &mut freed, self.conv_in_w, "conv_in.weight");
        free_t(gpu, &mut freed, self.conv_in_b, "conv_in.bias");
        for (i, r) in self.mid_resnet.into_iter().enumerate() {
            free_resnet(gpu, &mut freed, r, &format!("mid_resnet[{i}]"));
        }
        if let Some(a) = self.mid_attn {
            free_mid_attn(gpu, &mut freed, a);
        }
        for (i, b) in self.up_blocks.into_iter().enumerate() {
            for (j, r) in b.resnets.into_iter().enumerate() {
                free_resnet(gpu, &mut freed, r, &format!("up[{i}].resnet[{j}]"));
            }
            if let Some(s) = b.upsample_w {
                free_t(gpu, &mut freed, s, &format!("up[{i}].upsample.weight"));
            }
            if let Some(s) = b.upsample_b {
                free_t(gpu, &mut freed, s, &format!("up[{i}].upsample.bias"));
            }
        }
        free_t(gpu, &mut freed, self.conv_norm_out_w, "norm_out.weight");
        free_t(gpu, &mut freed, self.conv_norm_out_b, "norm_out.bias");
        free_t(gpu, &mut freed, self.conv_out_w, "conv_out.weight");
        free_t(gpu, &mut freed, self.conv_out_b, "conv_out.bias");
        if let Some((w, b)) = self.post_quant_conv {
            free_t(gpu, &mut freed, w, "post_quant_conv.weight");
            free_t(gpu, &mut freed, b, "post_quant_conv.bias");
        }
        freed
    }
}

fn free_t(gpu: &mut Gpu, freed: &mut usize, t: GpuTensor, what: &str) {
    gpu.free_tensor(t)
        .unwrap_or_else(|e| panic!("vae gpu: free {what}: {e:?}"));
    *freed += 1;
}

fn free_resnet(gpu: &mut Gpu, freed: &mut usize, r: GpuVaeResnet, what: &str) {
    free_t(gpu, freed, r.norm1_w, &format!("{what}.norm1.weight"));
    free_t(gpu, freed, r.norm1_b, &format!("{what}.norm1.bias"));
    free_t(gpu, freed, r.conv1_w, &format!("{what}.conv1.weight"));
    free_t(gpu, freed, r.conv1_b, &format!("{what}.conv1.bias"));
    free_t(gpu, freed, r.norm2_w, &format!("{what}.norm2.weight"));
    free_t(gpu, freed, r.norm2_b, &format!("{what}.norm2.bias"));
    free_t(gpu, freed, r.conv2_w, &format!("{what}.conv2.weight"));
    free_t(gpu, freed, r.conv2_b, &format!("{what}.conv2.bias"));
    if let Some(s) = r.shortcut_w {
        free_t(gpu, freed, s, &format!("{what}.shortcut.weight"));
    }
    if let Some(s) = r.shortcut_b {
        free_t(gpu, freed, s, &format!("{what}.shortcut.bias"));
    }
}

fn free_mid_attn(gpu: &mut Gpu, freed: &mut usize, a: GpuMidAttn) {
    free_t(gpu, freed, a.group_norm_w, "mid_attn.norm.weight");
    free_t(gpu, freed, a.group_norm_b, "mid_attn.norm.bias");
    free_t(gpu, freed, a.q_w, "mid_attn.q.weight");
    free_t(gpu, freed, a.q_b, "mid_attn.q.bias");
    free_t(gpu, freed, a.k_w, "mid_attn.k.weight");
    free_t(gpu, freed, a.k_b, "mid_attn.k.bias");
    free_t(gpu, freed, a.v_w, "mid_attn.v.weight");
    free_t(gpu, freed, a.v_b, "mid_attn.v.bias");
    free_t(gpu, freed, a.out_w, "mid_attn.proj.weight");
    free_t(gpu, freed, a.out_b, "mid_attn.proj.bias");
}

impl GpuVaeEncoderWeights {
    /// Upload every encoder weight of a loaded host [`VaeEncoderWeights`].
    ///
    /// Conv weights stay f32: the encode takes the direct per-thread conv
    /// route unconditionally (see the type docs), so there is no f16 cast and
    /// no `all_convs_gemm_ready` decision to make. Failure frees what it
    /// uploaded, same [`UploadLedger`] discipline as the decoder.
    pub fn from_host(gpu: &mut Gpu, host: &VaeEncoderWeights) -> Result<Self, String> {
        let mut led = UploadLedger::default();
        match Self::upload_into(gpu, &mut led, host) {
            Ok(slots) => Ok(GpuVaeEncoderWeights {
                config: host.config.clone(),
                conv_in_w: led.take(slots.conv_in_w),
                conv_in_b: led.take(slots.conv_in_b),
                down_blocks: slots
                    .down_blocks
                    .into_iter()
                    .map(|b| b.take(&mut led))
                    .collect(),
                mid_resnet: slots
                    .mid_resnet
                    .into_iter()
                    .map(|r| r.take(&mut led))
                    .collect(),
                mid_attn: slots.mid_attn.map(|a| a.take(&mut led)),
                conv_norm_out_w: led.take(slots.conv_norm_out_w),
                conv_norm_out_b: led.take(slots.conv_norm_out_b),
                conv_out_w: led.take(slots.conv_out_w),
                conv_out_b: led.take(slots.conv_out_b),
                quant_conv: slots.quant_conv.map(|(w, b)| (led.take(w), led.take(b))),
            }),
            Err(e) => {
                let freed = led.free_partial(gpu);
                Err(format!("{e} [freed {freed} partially-uploaded tensors]"))
            }
        }
    }

    fn upload_into(
        gpu: &mut Gpu,
        led: &mut UploadLedger,
        host: &VaeEncoderWeights,
    ) -> Result<SlotEncoder, String> {
        // f32 everywhere: the encode's `Run` is built with `conv_gemm=false`.
        const F16: bool = false;
        let mut down_blocks = Vec::with_capacity(host.down_blocks.len());
        for (i, b) in host.down_blocks.iter().enumerate() {
            let mut resnets = Vec::with_capacity(b.resnets.len());
            for (j, r) in b.resnets.iter().enumerate() {
                resnets.push(resnet_from_host(
                    gpu,
                    led,
                    r,
                    &format!("down[{i}].resnet[{j}]"),
                    F16,
                )?);
            }
            let (dw, db) = match (&b.downsample_w, &b.downsample_b) {
                (Some(w), Some(bb)) => (
                    Some(up_conv(
                        gpu,
                        led,
                        w,
                        &format!("down[{i}].downsample.weight"),
                        F16,
                    )?),
                    Some(up(gpu, led, bb, &format!("down[{i}].downsample.bias"))?),
                ),
                (None, None) => (None, None),
                _ => return Err(format!("vae gpu: down[{i}]: half-present downsample")),
            };
            down_blocks.push(SlotDownBlock {
                channels: b.channels,
                resnets,
                downsample_w: dw,
                downsample_b: db,
            });
        }
        let mut mid_resnet = Vec::with_capacity(host.mid_resnet.len());
        for (i, r) in host.mid_resnet.iter().enumerate() {
            mid_resnet.push(resnet_from_host(
                gpu,
                led,
                r,
                &format!("enc_mid_resnet[{i}]"),
                F16,
            )?);
        }
        let mid_attn = match &host.mid_attn {
            None => None,
            Some(a) => Some(mid_attn_from_host(gpu, led, a, F16)?),
        };
        let quant_conv = match &host.quant_conv {
            None => None,
            Some((w, b)) => Some((
                up_conv(gpu, led, w, "quant_conv.weight", F16)?,
                up(gpu, led, b, "quant_conv.bias")?,
            )),
        };
        Ok(SlotEncoder {
            conv_in_w: up_conv(gpu, led, &host.conv_in_w, "enc_conv_in.weight", F16)?,
            conv_in_b: up(gpu, led, &host.conv_in_b, "enc_conv_in.bias")?,
            down_blocks,
            mid_resnet,
            mid_attn,
            conv_norm_out_w: up(gpu, led, &host.conv_norm_out_w, "enc_norm_out.weight")?,
            conv_norm_out_b: up(gpu, led, &host.conv_norm_out_b, "enc_norm_out.bias")?,
            conv_out_w: up_conv(gpu, led, &host.conv_out_w, "enc_conv_out.weight", F16)?,
            conv_out_b: up(gpu, led, &host.conv_out_b, "enc_conv_out.bias")?,
            quant_conv,
        })
    }

    /// Return every GPU buffer to the pool. Exhaustive by construction.
    pub fn free_gpu(self, gpu: &mut Gpu) -> usize {
        let mut freed = 0usize;
        free_t(gpu, &mut freed, self.conv_in_w, "enc_conv_in.weight");
        free_t(gpu, &mut freed, self.conv_in_b, "enc_conv_in.bias");
        for (i, b) in self.down_blocks.into_iter().enumerate() {
            for (j, r) in b.resnets.into_iter().enumerate() {
                free_resnet(gpu, &mut freed, r, &format!("down[{i}].resnet[{j}]"));
            }
            if let Some(s) = b.downsample_w {
                free_t(gpu, &mut freed, s, &format!("down[{i}].downsample.weight"));
            }
            if let Some(s) = b.downsample_b {
                free_t(gpu, &mut freed, s, &format!("down[{i}].downsample.bias"));
            }
        }
        for (i, r) in self.mid_resnet.into_iter().enumerate() {
            free_resnet(gpu, &mut freed, r, &format!("enc_mid_resnet[{i}]"));
        }
        if let Some(a) = self.mid_attn {
            free_mid_attn(gpu, &mut freed, a);
        }
        free_t(gpu, &mut freed, self.conv_norm_out_w, "enc_norm_out.weight");
        free_t(gpu, &mut freed, self.conv_norm_out_b, "enc_norm_out.bias");
        free_t(gpu, &mut freed, self.conv_out_w, "enc_conv_out.weight");
        free_t(gpu, &mut freed, self.conv_out_b, "enc_conv_out.bias");
        if let Some((w, b)) = self.quant_conv {
            free_t(gpu, &mut freed, w, "quant_conv.weight");
            free_t(gpu, &mut freed, b, "quant_conv.bias");
        }
        freed
    }
}

fn mid_attn_from_host(
    gpu: &mut Gpu,
    led: &mut UploadLedger,
    a: &MidAttn,
    f16: bool,
) -> Result<SlotMidAttn, String> {
    Ok(SlotMidAttn {
        group_norm_w: up(gpu, led, &a.group_norm_w, "mid_attn.norm.weight")?,
        group_norm_b: up(gpu, led, &a.group_norm_b, "mid_attn.norm.bias")?,
        q_w: up_conv(gpu, led, &a.q_w, "mid_attn.q.weight", f16)?,
        q_b: up(gpu, led, &a.q_b, "mid_attn.q.bias")?,
        k_w: up_conv(gpu, led, &a.k_w, "mid_attn.k.weight", f16)?,
        k_b: up(gpu, led, &a.k_b, "mid_attn.k.bias")?,
        v_w: up_conv(gpu, led, &a.v_w, "mid_attn.v.weight", f16)?,
        v_b: up(gpu, led, &a.v_b, "mid_attn.v.bias")?,
        out_w: up_conv(gpu, led, &a.out_w, "mid_attn.proj.weight", f16)?,
        out_b: up(gpu, led, &a.out_b, "mid_attn.proj.bias")?,
    })
}

// ─────────────── GPU decode forward ──────────────────────────────────────

/// Named stage dumps of the GPU decode (host-downloaded), mirroring the CPU
/// [`crate::vae::VaeStages`] so the two can be diffed stage-by-stage.
pub struct GpuVaeStages {
    pub conv_in: Vec<f32>,
    pub mid_r0: Vec<f32>,
    pub mid_attn: Vec<f32>,
    pub mid_r1: Vec<f32>,
    pub up: Vec<f32>,
    pub out: Vec<f32>,
    pub out_h: usize,
    pub out_w: usize,
}

/// Byte budget for one conv's f16 column matrix (`HIPFIRE_VAE_IM2COL_MB`).
///
/// The whole-image im2col of a 128-channel 1024x1024 conv is
/// `1024*1024*1152*2` = 2.4 GB, and there are several of them per decode.
/// A device that is already holding the resident transformer serves a
/// multi-GB allocation very differently from the empty device the VAE parity
/// harness runs on, so the conv is staged over horizontal bands and the
/// column matrix is capped here instead. 256 MB still gives the GEMM a batch
/// of ~100k rows at 1024x1024, which is far past the point where batch size
/// matters to it.
const IM2COL_BUDGET_MB_DEFAULT: usize = 256;

struct Run<'a> {
    gpu: &'a mut Gpu,
    groups: usize,
    eps: f32,
    /// true (default): 3x3 convs route im2col -> WMMA f16 GEMM -> transpose,
    /// reading the f16-resident conv weights directly. False when the weights
    /// were uploaded f32 for the direct one-thread-per-output kernel
    /// (`HIPFIRE_VAE_CONV=direct`, or a decoder with a K the GEMM can't take)
    /// — it tracks [`GpuVaeDecoderWeights::conv_w_f16`], so the route and the
    /// resident dtype can never disagree.
    conv_gemm: bool,
    /// Fold the SiLU that follows every non-attention GroupNorm into the
    /// norm kernel. `HIPFIRE_VAE_FUSE_NORM=0` restores the separate pair.
    fuse_norm: bool,
    /// Per-conv f16 column-matrix budget in bytes.
    im2col_budget: usize,
    prof: bool,
    started: std::time::Instant,
    times: std::collections::BTreeMap<String, (u64, std::time::Duration)>,
}

impl<'a> Run<'a> {
    fn new(gpu: &'a mut Gpu, groups: usize, conv_gemm: bool) -> Self {
        let prof = hipfire_config::developer_var("HIPFIRE_VAE_PROFILE").map_or(false, |v| v != "0");
        let fuse_norm =
            hipfire_config::developer_var("HIPFIRE_VAE_FUSE_NORM").map_or(true, |v| v != "0");
        let budget_mb = hipfire_config::developer_var("HIPFIRE_VAE_IM2COL_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|mb| *mb > 0)
            .unwrap_or(IM2COL_BUDGET_MB_DEFAULT);
        Self {
            gpu,
            groups,
            eps: 1e-6,
            conv_gemm,
            fuse_norm,
            im2col_budget: budget_mb * 1024 * 1024,
            prof,
            started: std::time::Instant::now(),
            times: std::collections::BTreeMap::new(),
        }
    }

    fn sync(&mut self) {
        let _ = self.gpu.hip.device_synchronize();
    }

    /// Time one kernel launch when profiling. Syncs before/after so the
    /// measurement isolates this launch; with profiling off it is a passthrough
    /// with zero synchronization.
    fn timed<T>(
        &mut self,
        label: &str,
        launch: impl FnOnce(&mut Gpu) -> Result<T, String>,
    ) -> Result<T, String> {
        if !self.prof {
            return launch(&mut *self.gpu);
        }
        self.sync();
        let t0 = std::time::Instant::now();
        let out = launch(&mut *self.gpu)?;
        self.sync();
        let dt = t0.elapsed();
        let e = self
            .times
            .entry(label.to_string())
            .or_insert((0, std::time::Duration::ZERO));
        e.0 += 1;
        e.1 += dt;
        Ok(out)
    }

    fn report(&self) {
        if !self.prof {
            return;
        }
        let total: std::time::Duration = self.times.values().map(|(_, d)| *d).sum();
        let wall = self.started.elapsed();
        // Wall vs. kernel is the diagnostic that separates a slow kernel from
        // a slow allocator: the timed launches are sync-bracketed, so
        // anything in the gap is alloc/free, upload/download, or driver
        // work — not compute.
        eprintln!(
            "vae gpu profile (wall {:?}, total kernel time {:?}, off-kernel {:?})",
            wall,
            total,
            wall.saturating_sub(total)
        );
        let mut rows: Vec<(&String, &(u64, std::time::Duration))> = self.times.iter().collect();
        rows.sort_by(|a, b| b.1 .1.cmp(&a.1 .1));
        for (label, (count, dur)) in rows {
            let pct = if total.as_secs_f64() > 0.0 {
                100.0 * dur.as_secs_f64() / total.as_secs_f64()
            } else {
                0.0
            };
            eprintln!(
                "  {label:<14} {count:>4}x  {:>10.3} ms  {pct:5.1}%",
                dur.as_secs_f64() * 1000.0
            );
        }
    }

    fn alloc(&mut self, shape: &[usize]) -> Result<GpuTensor, String> {
        self.gpu
            .alloc_tensor(shape, DType::F32)
            .map_err(|e| format!("vae gpu: alloc {shape:?}: {e:?}"))
    }

    fn alloc_f16(&mut self, shape: &[usize]) -> Result<GpuTensor, String> {
        self.gpu
            .alloc_tensor(shape, DType::F16)
            .map_err(|e| format!("vae gpu: alloc f16 {shape:?}: {e:?}"))
    }

    fn free(&mut self, t: GpuTensor, what: &str) -> Result<(), String> {
        self.gpu
            .free_tensor(t)
            .map_err(|e| format!("vae gpu: free {what}: {e:?}"))
    }

    fn silu(&mut self, x: &GpuTensor) -> Result<GpuTensor, String> {
        let y = self.alloc(&x.shape)?;
        self.timed("silu", |gpu| {
            gpu.silu_f32(x, &y)
                .map_err(|e| format!("vae gpu: silu: {e:?}"))
        })?;
        Ok(y)
    }

    fn groupnorm(
        &mut self,
        x: &GpuTensor,
        c: usize,
        hw: usize,
        gamma: &GpuTensor,
        beta: &GpuTensor,
    ) -> Result<GpuTensor, String> {
        let y = self.alloc(&x.shape)?;
        let groups = self.groups;
        let eps = self.eps;
        self.timed("groupnorm", |gpu| {
            gpu.vae_groupnorm_f32(x, gamma, beta, &y, c, hw, groups, eps)
                .map_err(|e| format!("vae gpu: groupnorm c={c} hw={hw}: {e:?}"))
        })?;
        Ok(y)
    }

    /// GroupNorm immediately followed by SiLU — the pair every VAE resnet and
    /// the decoder tail run. Fused into one kernel by default: the separate
    /// SiLU is a whole extra read+write pass over a tensor that reaches
    /// 512 MB at the 1024x1024 tail, and it computes the same f32 expression
    /// on the same value, so the fused result is bit-identical.
    fn groupnorm_silu(
        &mut self,
        x: &GpuTensor,
        c: usize,
        hw: usize,
        gamma: &GpuTensor,
        beta: &GpuTensor,
    ) -> Result<GpuTensor, String> {
        if !self.fuse_norm {
            let n = self.groupnorm(x, c, hw, gamma, beta)?;
            let a = self.silu(&n)?;
            self.free(n, "groupnorm_silu.norm")?;
            return Ok(a);
        }
        let y = self.alloc(&x.shape)?;
        let groups = self.groups;
        let eps = self.eps;
        self.timed("gnorm_silu", |gpu| {
            gpu.vae_groupnorm_silu_f32(x, gamma, beta, &y, c, hw, groups, eps)
                .map_err(|e| format!("vae gpu: groupnorm+silu c={c} hw={hw}: {e:?}"))
        })?;
        Ok(y)
    }

    /// 3x3 stride-1 pad-1 conv. Default route is im2col -> tuned WMMA f16
    /// GEMM (fused bias) -> transpose back to channel-major: the naive f32
    /// one-thread-per-output kernel spent two thirds of the decode. The
    /// GEMM computes `y[p, co] = bias[co] + sum_k im2col[p, k] * w[co, k]`,
    /// i.e. position-major `[hw][c_out]`, hence the transpose. `w` is already
    /// the f16 resident weight (see [`GpuVaeDecoderWeights::conv_w_f16`]).
    ///
    /// The image is walked in horizontal bands so the f16 column matrix stays
    /// inside [`Run::im2col_budget`] instead of reaching 2.4 GB in one
    /// allocation. Each band's GEMM writes straight into its slice of the
    /// channel-major output through the banded transpose.
    ///
    /// Banding is exact in the data movement — the im2col taps still read the
    /// whole image, so a band's first and last rows see their real
    /// neighbours, not padding, and the banded transpose only changes where
    /// the same values land. It is NOT exact by construction in the GEMM:
    /// `gemm_f16_x_f16_wmma_lds_auto` picks its macro-tile from
    /// `lds_tile_for(arch, m, batch, cu)` using the band's row count, and
    /// candidate tiles differ in k-step, so a different band count can change
    /// the f16 accumulation order. Bit-exactness across band counts is
    /// therefore a measured property of the shapes in play, not an invariant
    /// — `test_vae_lds`' "conv banded" subtest is the evidence, and it
    /// reports the observed deviation rather than assuming zero.
    ///
    /// K = c_in*9 must be a multiple of 16 (true for every real FLUX conv:
    /// 144 / 1152 / 2304 / 4608); a decoder that misses it takes the direct
    /// route for ALL its convs, decided once at upload.
    fn conv3x3(
        &mut self,
        x: &GpuTensor,
        w: &GpuTensor,
        bias: &GpuTensor,
        c_in: usize,
        c_out: usize,
        h: usize,
        wdt: usize,
    ) -> Result<GpuTensor, String> {
        let hw = h * wdt;
        let k = c_in * 9;
        if !self.conv_gemm {
            let y = self.alloc(&[c_out * hw])?;
            self.timed("conv3x3", |gpu| {
                gpu.vae_conv3x3_f32(x, w, bias, &y, c_in, c_out, h, wdt)
                    .map_err(|e| format!("vae gpu: conv3x3 {c_in}->{c_out} @ {h}x{wdt}: {e:?}"))
            })?;
            return Ok(y);
        }
        if k % 16 != 0 {
            return Err(format!(
                "vae gpu: conv3x3 {c_in}->{c_out}: K={k} is not a multiple of 16, but the \
                 conv weights are f16-resident for the GEMM route; re-upload with \
                 HIPFIRE_VAE_CONV=direct"
            ));
        }
        // Rows per band: whole rows only, at least one, capped so the f16
        // column matrix fits the budget.
        let row_bytes = wdt * k * 2;
        let band = (self.im2col_budget / row_bytes.max(1)).clamp(1, h);
        let y = self.alloc(&[c_out * hw])?;
        let cols = self.alloc_f16(&[band * wdt, k])?;
        let pos = self.alloc(&[band * wdt, c_out])?;
        let mut y0 = 0usize;
        while y0 < h {
            let rows = band.min(h - y0);
            let m = rows * wdt;
            self.timed("im2col", |gpu| {
                gpu.vae_im2col_f16_band(x, &cols, c_in, h, wdt, y0, rows)
                    .map_err(|e| format!("vae gpu: im2col {c_in} @ {h}x{wdt}: {e:?}"))
            })?;
            self.timed("gemm", |gpu| {
                // Operand order mirrors `Gpuf::gemm_pre`: weights first
                // (`[c_out, k]`), activations second, dims (c_out, k, m) — the
                // output is position-major `[m, c_out]` and the fused bias is
                // indexed by the FIRST operand's rows (c_out).
                if k % 64 == 0 {
                    gpu.gemm_f16_x_f16_wmma_lds_auto(w, &cols, &pos, Some(bias), c_out, k, m)
                        .map_err(|e| {
                            format!("vae gpu: conv gemm {c_in}->{c_out} @ {h}x{wdt}: {e:?}")
                        })
                } else {
                    // Ragged K (conv_in: 16ch -> K=144): the 16-step kernel has
                    // no fused bias, so bias_add follows.
                    gpu.gemm_f16_x_f16_wmma(w, &cols, &pos, c_out, k, m)
                        .map_err(|e| {
                            format!("vae gpu: conv gemm16 {c_in}->{c_out} @ {h}x{wdt}: {e:?}")
                        })
                }
            })?;
            if k % 64 != 0 {
                self.timed("bias_add", |gpu| {
                    gpu.bias_add_f32(&pos, bias, m, c_out)
                        .map_err(|e| format!("vae gpu: conv bias: {e:?}"))
                })?;
            }
            let off = y0 * wdt;
            self.timed("transpose", |gpu| {
                gpu.vae_transpose_f32_banded(&pos, &y, m, c_out, hw, off)
                    .map_err(|e| format!("vae gpu: conv transpose {c_out} @ {h}x{wdt}: {e:?}"))
            })?;
            y0 += rows;
        }
        self.free(cols, "conv.im2col")?;
        self.free(pos, "conv.pos")?;
        Ok(y)
    }

    /// 3x3 STRIDE-2 conv with the diffusers `Downsample2D` asymmetric pad
    /// (0,1,0,1) — the VAE encoder's per-block downsampler. Output is
    /// `[c_out][h/2][wdt/2]`.
    ///
    /// Direct route only. There is no GEMM variant because the encoder always
    /// builds its [`Run`] with `conv_gemm = false` (it runs once per reference
    /// image), so f16-resident weights would be read as f32 garbage here — the
    /// guard turns that into an error instead of a silently wrong latent.
    fn conv3x3_s2(
        &mut self,
        x: &GpuTensor,
        w: &GpuTensor,
        bias: &GpuTensor,
        c_in: usize,
        c_out: usize,
        h: usize,
        wdt: usize,
    ) -> Result<GpuTensor, String> {
        if self.conv_gemm {
            return Err(format!(
                "vae gpu: conv3x3_s2 {c_in}->{c_out}: no GEMM route exists for the \
                 stride-2 downsampler, but this Run has conv_gemm=true (f16-resident \
                 weights); build the encoder Run with conv_gemm=false"
            ));
        }
        let y = self.alloc(&[c_out * (h / 2) * (wdt / 2)])?;
        self.timed("conv3x3_s2", |gpu| {
            gpu.vae_conv3x3_s2_f32(x, w, bias, &y, c_in, c_out, h, wdt)
                .map_err(|e| format!("vae gpu: conv3x3_s2 {c_in}->{c_out} @ {h}x{wdt}: {e:?}"))
        })?;
        Ok(y)
    }

    /// Channel-major f32 `[c_in][n]` -> position-major f16 `[n][c_in]`, the
    /// GEMM operand form. One fused transpose+cast, shareable across several
    /// GEMMs (the mid-attn q/k/v projections all read the same normed map).
    fn transpose_cast_f16(
        &mut self,
        x: &GpuTensor,
        c_in: usize,
        n: usize,
    ) -> Result<GpuTensor, String> {
        let xt = self.alloc_f16(&[n, c_in])?;
        self.timed("xcast", |gpu| {
            gpu.vae_transpose_cast_f16(x, &xt, c_in, n)
                .map_err(|e| format!("vae gpu: transpose cast {c_in}x{n}: {e:?}"))
        })?;
        Ok(xt)
    }

    /// 1x1 conv GEMM core: `xf16` is the prepared position-major f16
    /// `[n][c_in]` operand and `w` the f16-resident `[c_out][c_in]` weight;
    /// returns the position-major f32 `[n][c_out]` result with the bias
    /// fused. K = c_in, so real VAE channels (128 and up, all %64) take the
    /// LDS macro-tile kernel.
    fn conv1x1_gemm(
        &mut self,
        xf16: &GpuTensor,
        w: &GpuTensor,
        bias: &GpuTensor,
        c_in: usize,
        c_out: usize,
        n: usize,
    ) -> Result<GpuTensor, String> {
        let pos = self.alloc(&[n, c_out])?;
        self.timed("gemm", |gpu| {
            if c_in % 64 == 0 {
                gpu.gemm_f16_x_f16_wmma_lds_auto(w, xf16, &pos, Some(bias), c_out, c_in, n)
                    .map_err(|e| format!("vae gpu: conv1x1 gemm {c_in}->{c_out}: {e:?}"))
            } else {
                gpu.gemm_f16_x_f16_wmma(w, xf16, &pos, c_out, c_in, n)
                    .map_err(|e| format!("vae gpu: conv1x1 gemm16 {c_in}->{c_out}: {e:?}"))
            }
        })?;
        if c_in % 64 != 0 {
            self.timed("bias_add", |gpu| {
                gpu.bias_add_f32(&pos, bias, n, c_out)
                    .map_err(|e| format!("vae gpu: conv1x1 bias: {e:?}"))
            })?;
        }
        Ok(pos)
    }

    /// 1x1 conv (per-pixel linear). Default route is the WMMA GEMM: the
    /// input is prepared position-major f16 (a fused transpose+cast when it
    /// arrives channel-major), the GEMM fuses the bias, and a channel-major
    /// result (the resnet shortcuts) transposes once on the way out. The
    /// naive per-thread kernel is the `HIPFIRE_VAE_CONV=direct` fallback and
    /// the route for ragged c_in.
    fn conv1x1(
        &mut self,
        x: &GpuTensor,
        w: &GpuTensor,
        bias: &GpuTensor,
        c_in: usize,
        c_out: usize,
        n: usize,
        in_pos_major: bool,
        out_pos_major: bool,
    ) -> Result<GpuTensor, String> {
        if !self.conv_gemm {
            let y = self.alloc(&[c_out * n])?;
            self.timed("conv1x1", |gpu| {
                gpu.vae_conv1x1_f32(x, w, bias, &y, c_in, c_out, n, in_pos_major, out_pos_major)
                    .map_err(|e| format!("vae gpu: conv1x1 {c_in}->{c_out} n={n}: {e:?}"))
            })?;
            return Ok(y);
        }
        if c_in % 16 != 0 {
            return Err(format!(
                "vae gpu: conv1x1 {c_in}->{c_out}: c_in is not a multiple of 16, but the \
                 conv weights are f16-resident for the GEMM route; re-upload with \
                 HIPFIRE_VAE_CONV=direct"
            ));
        }
        let held_xf;
        let xf16: &GpuTensor = if in_pos_major {
            let cast = self.alloc_f16(&[n, c_in])?;
            self.timed("xcast", |gpu| {
                gpu.cast_f32_to_f16(x, &cast)
                    .map_err(|e| format!("vae gpu: conv1x1 x cast: {e:?}"))
            })?;
            held_xf = cast;
            &held_xf
        } else {
            held_xf = self.transpose_cast_f16(x, c_in, n)?;
            &held_xf
        };
        let pos = self.conv1x1_gemm(xf16, w, bias, c_in, c_out, n)?;
        self.free(held_xf, "conv1x1.xf16")?;
        if out_pos_major {
            return Ok(pos);
        }
        let y = self.alloc(&[c_out * n])?;
        self.timed("transpose", |gpu| {
            gpu.vae_transpose_f32(&pos, &y, n, c_out)
                .map_err(|e| format!("vae gpu: conv1x1 transpose {c_out}x{n}: {e:?}"))
        })?;
        self.free(pos, "conv1x1.pos")?;
        Ok(y)
    }

    fn add(&mut self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor, String> {
        let y = self.alloc(&a.shape)?;
        self.timed("add", |gpu| {
            gpu.add_f32(a, b, &y)
                .map_err(|e| format!("vae gpu: add: {e:?}"))
        })?;
        Ok(y)
    }

    /// One taming `ResnetBlock` on the GPU, mirroring the CPU
    /// [`crate::vae`] `resnet_forward`: norm1 -> SiLU -> conv1 -> norm2 ->
    /// SiLU -> conv2 -> + residual (identity or 1x1 shortcut). Consumes `x`.
    fn resnet(
        &mut self,
        x: GpuTensor,
        h: usize,
        wdt: usize,
        r: &GpuVaeResnet,
    ) -> Result<GpuTensor, String> {
        let hw = h * wdt;
        let a1 = self.groupnorm_silu(&x, r.in_ch, hw, &r.norm1_w, &r.norm1_b)?;
        let c1 = self.conv3x3(&a1, &r.conv1_w, &r.conv1_b, r.in_ch, r.out_ch, h, wdt)?;
        self.free(a1, "resnet.act1")?;
        let a2 = self.groupnorm_silu(&c1, r.out_ch, hw, &r.norm2_w, &r.norm2_b)?;
        // `c1` was leaked here before: the block held it to the end of the
        // decode, and at 1024x1024 that is 512 MB of pool never handed back
        // for the next resnet to reuse.
        self.free(c1, "resnet.conv1")?;
        let c2 = self.conv3x3(&a2, &r.conv2_w, &r.conv2_b, r.out_ch, r.out_ch, h, wdt)?;
        self.free(a2, "resnet.act2")?;
        let residual = if r.in_ch == r.out_ch {
            x
        } else {
            let sw = r
                .shortcut_w
                .as_ref()
                .expect("channel change needs shortcut weights");
            let sb = r
                .shortcut_b
                .as_ref()
                .expect("channel change needs shortcut bias");
            let s = self.conv1x1(&x, sw, sb, r.in_ch, r.out_ch, hw, false, false)?;
            self.free(x, "resnet.input")?;
            s
        };
        let y = self.add(&residual, &c2)?;
        self.free(residual, "resnet.residual")?;
        self.free(c2, "resnet.conv2")?;
        Ok(y)
    }

    /// taming `Upsample` on the GPU: nearest 2x then a 3x3 pad-1 conv.
    /// Consumes `x`.
    fn upsample(
        &mut self,
        x: GpuTensor,
        c: usize,
        h: usize,
        wdt: usize,
        cw: &GpuTensor,
        cb: &GpuTensor,
    ) -> Result<(GpuTensor, usize, usize), String> {
        let (oh, ow) = (2 * h, 2 * wdt);
        let up = self.alloc(&[c * oh * ow])?;
        self.timed("upsample2x", |gpu| {
            gpu.vae_upsample2x_f32(&x, &up, c, h, wdt)
                .map_err(|e| format!("vae gpu: upsample2x {c} @ {h}x{wdt}: {e:?}"))
        })?;
        self.free(x, "upsample.input")?;
        let y = self.conv3x3(&up, cw, cb, c, c, oh, ow)?;
        self.free(up, "upsample.nearest")?;
        Ok((y, oh, ow))
    }

    /// Mid-block self-attention on the GPU, mirroring the CPU
    /// `mid_attn_forward`: groupnorm, 1x1 q/k/v to position-major `[n][c]`,
    /// single-head scaled dot-product softmax attention, 1x1 proj_out, and a
    /// fused transpose-residual back to channel-major. Consumes `x`.
    ///
    /// GEMM route (default): one shared transpose+cast prepares the normed
    /// map for the q/k/v projections; scores and ctx are WMMA GEMMs too
    /// (`scores = k.q^T` with k first so the layout lands untransposed,
    /// `ctx = probs . v` with channel-major v as the A operand). The naive
    /// kernels stay as the `HIPFIRE_VAE_CONV=direct` / ragged-c fallback.
    fn mid_attn(
        &mut self,
        x: GpuTensor,
        c: usize,
        h: usize,
        wdt: usize,
        a: &GpuMidAttn,
    ) -> Result<GpuTensor, String> {
        let n = h * wdt;
        let normed = self.groupnorm(&x, c, n, &a.group_norm_w, &a.group_norm_b)?;
        let scale = 1.0 / (c as f32).sqrt();
        let gemm_route = self.conv_gemm && c % 16 == 0;
        let (q, k, v) = if gemm_route {
            let nt = self.transpose_cast_f16(&normed, c, n)?;
            self.free(normed, "attn.normed")?;
            let q = self.conv1x1_gemm(&nt, &a.q_w, &a.q_b, c, c, n)?;
            let k = self.conv1x1_gemm(&nt, &a.k_w, &a.k_b, c, c, n)?;
            let v_pos = self.conv1x1_gemm(&nt, &a.v_w, &a.v_b, c, c, n)?;
            self.free(nt, "attn.nt")?;
            // The ctx GEMM wants v as channel-major f16 `[c][n]` (the A
            // operand), so the layout flip fuses the cast.
            let v = self.alloc_f16(&[c * n])?;
            self.timed("transpose", |gpu| {
                gpu.vae_transpose_cast_f16(&v_pos, &v, n, c)
                    .map_err(|e| format!("vae gpu: attn v transpose: {e:?}"))
            })?;
            self.free(v_pos, "attn.vpos")?;
            (q, k, v)
        } else {
            let q = self.conv1x1(&normed, &a.q_w, &a.q_b, c, c, n, false, true)?;
            let k = self.conv1x1(&normed, &a.k_w, &a.k_b, c, c, n, false, true)?;
            let v = self.conv1x1(&normed, &a.v_w, &a.v_b, c, c, n, false, true)?;
            self.free(normed, "attn.normed")?;
            (q, k, v)
        };
        let scores = self.alloc(&[n, n])?;
        if gemm_route {
            // scores[qp][kp] = scale * dot(q[qp], k[kp]). Kernel writes
            // `Y[b*M + m_out] = sum A[m_out]·X[b]`, so k first (m_out=kp),
            // q second (b=qp) lands `scale·S` untransposed at
            // `scores[qp*n + kp]` — the orientation the row-wise softmax and
            // ctx read. The scale folds into the q cast.
            let qf = self.alloc_f16(&[n, c])?;
            self.timed("qcast", |gpu| {
                gpu.vae_cast_scale_f16(&q, &qf, scale)
                    .map_err(|e| format!("vae gpu: attn q cast: {e:?}"))
            })?;
            let kf = self.alloc_f16(&[n, c])?;
            self.timed("kcast", |gpu| {
                gpu.cast_f32_to_f16(&k, &kf)
                    .map_err(|e| format!("vae gpu: attn k cast: {e:?}"))
            })?;
            self.timed("attn_scores", |gpu| {
                if c % 64 == 0 {
                    gpu.gemm_f16_x_f16_wmma_lds_auto(&kf, &qf, &scores, None, n, c, n)
                        .map_err(|e| format!("vae gpu: attn scores gemm n={n} c={c}: {e:?}"))
                } else {
                    gpu.gemm_f16_x_f16_wmma(&kf, &qf, &scores, n, c, n)
                        .map_err(|e| format!("vae gpu: attn scores gemm16 n={n} c={c}: {e:?}"))
                }
            })?;
            self.free(qf, "attn.qf")?;
            self.free(kf, "attn.kf")?;
        } else {
            self.timed("attn_scores", |gpu| {
                gpu.vae_attn_scores_f32(&q, &k, &scores, n, c, scale)
                    .map_err(|e| format!("vae gpu: attn scores n={n} c={c}: {e:?}"))
            })?;
        }
        self.free(q, "attn.q")?;
        self.free(k, "attn.k")?;
        self.timed("softmax", |gpu| {
            gpu.softmax_f32(&scores)
                .map_err(|e| format!("vae gpu: attn softmax n={n}: {e:?}"))
        })?;
        let ctx = if gemm_route && n % 16 == 0 {
            // ctx[p][ch] = sum_kp probs[p][kp] * v[kp][ch]: with channel-
            // major v `[c][n]` as A (m_out=ch, k=kp) and probs as X (b=p),
            // the GEMM lands position-major `[n][c]` directly. K = n.
            let pf = self.alloc_f16(&[n, n])?;
            self.timed("pcast", |gpu| {
                gpu.cast_f32_to_f16(&scores, &pf)
                    .map_err(|e| format!("vae gpu: attn probs cast: {e:?}"))
            })?;
            self.free(scores, "attn.scores")?;
            let ctx = self.alloc(&[n, c])?;
            self.timed("attn_ctx", |gpu| {
                if n % 64 == 0 {
                    gpu.gemm_f16_x_f16_wmma_lds_auto(&v, &pf, &ctx, None, c, n, n)
                        .map_err(|e| format!("vae gpu: attn ctx gemm n={n} c={c}: {e:?}"))
                } else {
                    gpu.gemm_f16_x_f16_wmma(&v, &pf, &ctx, c, n, n)
                        .map_err(|e| format!("vae gpu: attn ctx gemm16 n={n} c={c}: {e:?}"))
                }
            })?;
            self.free(pf, "attn.pf")?;
            ctx
        } else {
            let ctx = self.alloc(&[n, c])?;
            self.timed("attn_ctx", |gpu| {
                gpu.vae_attn_ctx_f32(&scores, &v, &ctx, n, c)
                    .map_err(|e| format!("vae gpu: attn ctx n={n} c={c}: {e:?}"))
            })?;
            self.free(scores, "attn.scores")?;
            ctx
        };
        self.free(v, "attn.v")?;
        let proj = self.conv1x1(&ctx, &a.out_w, &a.out_b, c, c, n, true, true)?;
        self.free(ctx, "attn.ctx")?;
        let y = self.alloc(&[c * n])?;
        self.timed("attn_residual", |gpu| {
            gpu.vae_attn_residual_f32(&x, &proj, &y, c, n)
                .map_err(|e| format!("vae gpu: attn residual n={n} c={c}: {e:?}"))
        })?;
        self.free(x, "attn.input")?;
        self.free(proj, "attn.proj")?;
        Ok(y)
    }
}

/// FLUX.2's 1x1 latent->latent `post_quant_conv`, applied to the uploaded
/// channel-major latent before `conv_in`. Consumes `z` and returns its
/// replacement when the weight is present.
///
/// **FLUX.1 has no `post_quant_conv`, and this returns `z` untouched** —
/// no allocation, no launch — so the FLUX.1 decode issues exactly the
/// sequence it always did.
fn post_quant(
    run: &mut Run<'_>,
    w: &GpuVaeDecoderWeights,
    z: GpuTensor,
    hw: usize,
) -> Result<GpuTensor, String> {
    let Some((pw, pb)) = &w.post_quant_conv else {
        return Ok(z);
    };
    let lc = w.config.latent_channels;
    let y = run.conv1x1(&z, pw, pb, lc, lc, hw, false, false)?;
    run.free(z, "post_quant.input")?;
    Ok(y)
}

/// GPU VAE decode with named stage downloads, mirroring the CPU
/// [`crate::vae::decode_stages`]. `z` is the VAE-space latent
/// `[latent_channels][h_in][w_in]` (already shifted/scaled by the caller).
pub fn gpu_decode_stages(
    gpu: &mut Gpu,
    w: &GpuVaeDecoderWeights,
    z: &[f32],
    h_in: usize,
    w_in: usize,
) -> Result<GpuVaeStages, String> {
    let cfg = &w.config;
    let block_in = cfg.block_out_channels[cfg.block_out_channels.len() - 1];
    let groups = cfg.norm_num_groups;
    let mut run = Run::new(gpu, groups, w.conv_w_f16);
    let download = |gpu: &Gpu, t: &GpuTensor, what: &str| -> Result<Vec<f32>, String> {
        gpu.download_f32(t)
            .map_err(|e| format!("vae gpu: download {what}: {e:?}"))
    };

    let zt = run
        .gpu
        .upload_f32(z, &[cfg.latent_channels * h_in * w_in])
        .map_err(|e| format!("vae gpu: upload latent: {e:?}"))?;
    let zt = post_quant(&mut run, w, zt, h_in * w_in)?;
    let mut hidden = run.conv3x3(
        &zt,
        &w.conv_in_w,
        &w.conv_in_b,
        cfg.latent_channels,
        block_in,
        h_in,
        w_in,
    )?;
    run.free(zt, "latent")?;
    let (mut h, mut wdt) = (h_in, w_in);
    let conv_in = download(run.gpu, &hidden, "conv_in")?;

    hidden = run.resnet(hidden, h, wdt, &w.mid_resnet[0])?;
    let mid_r0 = download(run.gpu, &hidden, "mid_r0")?;
    if let Some(a) = &w.mid_attn {
        hidden = run.mid_attn(hidden, block_in, h, wdt, a)?;
    }
    let mid_attn = download(run.gpu, &hidden, "mid_attn")?;
    hidden = run.resnet(hidden, h, wdt, &w.mid_resnet[1])?;
    let mid_r1 = download(run.gpu, &hidden, "mid_r1")?;

    for block in &w.up_blocks {
        for r in &block.resnets {
            hidden = run.resnet(hidden, h, wdt, r)?;
        }
        if let (Some(cw), Some(cb)) = (&block.upsample_w, &block.upsample_b) {
            let (up, oh, ow) = run.upsample(hidden, block.channels, h, wdt, cw, cb)?;
            hidden = up;
            h = oh;
            wdt = ow;
        }
    }
    let up = download(run.gpu, &hidden, "up")?;

    let first_ch = cfg.block_out_channels[0];
    let act = run.groupnorm_silu(
        &hidden,
        first_ch,
        h * wdt,
        &w.conv_norm_out_w,
        &w.conv_norm_out_b,
    )?;
    run.free(hidden, "up.hidden")?;
    let out_t = run.conv3x3(
        &act,
        &w.conv_out_w,
        &w.conv_out_b,
        first_ch,
        cfg.out_channels,
        h,
        wdt,
    )?;
    run.free(act, "out.act")?;
    let out = download(run.gpu, &out_t, "out")?;
    run.free(out_t, "out")?;

    run.report();
    Ok(GpuVaeStages {
        conv_in,
        mid_r0,
        mid_attn,
        mid_r1,
        up,
        out,
        out_h: h,
        out_w: wdt,
    })
}

/// GPU VAE decode without stage downloads (the pipeline path): returns the
/// decoded pixels `[out_channels][h][w]` and the output spatial dims.
pub fn gpu_decode(
    gpu: &mut Gpu,
    w: &GpuVaeDecoderWeights,
    z: &[f32],
    h_in: usize,
    w_in: usize,
) -> Result<(Vec<f32>, usize, usize), String> {
    let cfg = &w.config;
    let block_in = cfg.block_out_channels[cfg.block_out_channels.len() - 1];
    let mut run = Run::new(gpu, cfg.norm_num_groups, w.conv_w_f16);

    let zt = run
        .gpu
        .upload_f32(z, &[cfg.latent_channels * h_in * w_in])
        .map_err(|e| format!("vae gpu: upload latent: {e:?}"))?;
    let zt = post_quant(&mut run, w, zt, h_in * w_in)?;
    let mut hidden = run.conv3x3(
        &zt,
        &w.conv_in_w,
        &w.conv_in_b,
        cfg.latent_channels,
        block_in,
        h_in,
        w_in,
    )?;
    run.free(zt, "latent")?;
    let (mut h, mut wdt) = (h_in, w_in);

    hidden = run.resnet(hidden, h, wdt, &w.mid_resnet[0])?;
    if let Some(a) = &w.mid_attn {
        hidden = run.mid_attn(hidden, block_in, h, wdt, a)?;
    }
    hidden = run.resnet(hidden, h, wdt, &w.mid_resnet[1])?;

    for block in &w.up_blocks {
        for r in &block.resnets {
            hidden = run.resnet(hidden, h, wdt, r)?;
        }
        if let (Some(cw), Some(cb)) = (&block.upsample_w, &block.upsample_b) {
            let (up, oh, ow) = run.upsample(hidden, block.channels, h, wdt, cw, cb)?;
            hidden = up;
            h = oh;
            wdt = ow;
        }
    }

    let first_ch = cfg.block_out_channels[0];
    let act = run.groupnorm_silu(
        &hidden,
        first_ch,
        h * wdt,
        &w.conv_norm_out_w,
        &w.conv_norm_out_b,
    )?;
    run.free(hidden, "up.hidden")?;
    let out_t = run.conv3x3(
        &act,
        &w.conv_out_w,
        &w.conv_out_b,
        first_ch,
        cfg.out_channels,
        h,
        wdt,
    )?;
    run.free(act, "out.act")?;
    let out = run
        .gpu
        .download_f32(&out_t)
        .map_err(|e| format!("vae gpu: download out: {e:?}"))?;
    run.free(out_t, "out")?;
    run.report();
    Ok((out, h, wdt))
}

/// GPU VAE encode, mirroring the CPU [`crate::vae::encode`]: pixels
/// `[in_channels][h][wdt]` in `[-1, 1]` -> latent MEAN
/// `[latent_channels][h/8][wdt/8]`.
///
/// Like the CPU reference this is argmax sample mode — the encoder emits
/// `2*latent` moment channels and only the leading `latent` (the mean half)
/// are downloaded; the logvar half is never sampled from.
///
/// The [`Run`] is built with `conv_gemm = false` on purpose: the encoder
/// weights are f32-resident (see [`GpuVaeEncoderWeights`]) and it runs once
/// per reference image, so the direct per-thread conv kernels are the whole
/// route. That also keeps it numerically f32 end to end, which is why the
/// parity threshold against the CPU encode is 1e-4 and not the decoder's
/// f16-GEMM 5e-3.
pub fn gpu_encode(
    gpu: &mut Gpu,
    w: &GpuVaeEncoderWeights,
    x: &[f32],
    h: usize,
    wdt: usize,
) -> Result<Vec<f32>, String> {
    let cfg = &w.config;
    if x.len() != cfg.in_channels * h * wdt {
        return Err(format!(
            "vae gpu: encode input has {} elems, expected {} for [{}][{h}][{wdt}]",
            x.len(),
            cfg.in_channels * h * wdt,
            cfg.in_channels
        ));
    }
    let mut run = Run::new(gpu, cfg.norm_num_groups, false);

    let xt = run
        .gpu
        .upload_f32(x, &[cfg.in_channels * h * wdt])
        .map_err(|e| format!("vae gpu: upload pixels: {e:?}"))?;
    let (mut hh, mut ww) = (h, wdt);
    let mut ch = cfg.block_out_channels[0];
    let mut cur = run.conv3x3(&xt, &w.conv_in_w, &w.conv_in_b, cfg.in_channels, ch, hh, ww)?;
    run.free(xt, "enc.pixels")?;

    for block in &w.down_blocks {
        for r in &block.resnets {
            cur = run.resnet(cur, hh, ww, r)?;
            ch = r.out_ch;
        }
        if let (Some(dw), Some(db)) = (&block.downsample_w, &block.downsample_b) {
            let down = run.conv3x3_s2(&cur, dw, db, ch, ch, hh, ww)?;
            run.free(cur, "enc.down.input")?;
            cur = down;
            hh /= 2;
            ww /= 2;
        }
    }

    cur = run.resnet(cur, hh, ww, &w.mid_resnet[0])?;
    if let Some(a) = &w.mid_attn {
        cur = run.mid_attn(cur, ch, hh, ww, a)?;
    }
    cur = run.resnet(cur, hh, ww, &w.mid_resnet[1])?;

    let act = run.groupnorm_silu(&cur, ch, hh * ww, &w.conv_norm_out_w, &w.conv_norm_out_b)?;
    run.free(cur, "enc.mid")?;
    let two = 2 * cfg.latent_channels;
    let mut moments = run.conv3x3(&act, &w.conv_out_w, &w.conv_out_b, ch, two, hh, ww)?;
    run.free(act, "enc.out.act")?;
    if let Some((qw, qb)) = &w.quant_conv {
        let q = run.conv1x1(&moments, qw, qb, two, two, hh * ww, false, false)?;
        run.free(moments, "enc.moments")?;
        moments = q;
    }
    let all = run
        .gpu
        .download_f32(&moments)
        .map_err(|e| format!("vae gpu: download moments: {e:?}"))?;
    run.free(moments, "enc.moments")?;
    run.report();

    // mean = the leading `latent` channels of the channel-major moments.
    let mean_len = cfg.latent_channels * hh * ww;
    if all.len() < mean_len {
        return Err(format!(
            "vae gpu: encode moments have {} elems, need {mean_len} for the mean half",
            all.len()
        ));
    }
    Ok(all[..mean_len].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ledger's slot bookkeeping is pure (no device calls), so it is
    /// testable without a GPU: `record` hands out stable indices, `take` moves
    /// a tensor out of the ledger, and `outstanding` reports exactly what
    /// `free_partial` would have to return to the pool on an error path.
    #[test]
    fn upload_ledger_tracks_outstanding_tensors() {
        let mut led = UploadLedger::default();
        assert_eq!(led.outstanding(), 0);

        let a = led.record(GpuTensor::null_for_test());
        let b = led.record(GpuTensor::null_for_test());
        let c = led.record(GpuTensor::null_for_test());
        assert_eq!((a, b, c), (0, 1, 2), "slots are handed out in order");
        assert_eq!(led.outstanding(), 3);

        // Assembling the finished structure empties the ledger, so a later
        // `free_partial` cannot double-free what the caller now owns.
        let _t = led.take(b);
        assert_eq!(led.outstanding(), 2);
        let _t = led.take(a);
        let _t = led.take(c);
        assert_eq!(
            led.outstanding(),
            0,
            "a fully-assembled decoder leaves nothing for the cleanup path"
        );
    }

    #[test]
    #[should_panic(expected = "taken twice")]
    fn upload_ledger_rejects_a_double_take() {
        let mut led = UploadLedger::default();
        let s = led.record(GpuTensor::null_for_test());
        let _t = led.take(s);
        let _t = led.take(s);
    }
}
