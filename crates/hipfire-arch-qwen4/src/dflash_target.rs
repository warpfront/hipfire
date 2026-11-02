// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen4 target side of the DFlash2 drafter (`hipfire_runtime::dflash2_spec`):
//! the drafter's target features (each listed trunk layer's completed output
//! averaged over the hyper-connection streams, captured between the
//! forward's steps), the raw embedding rows of its noise block, and the
//! target head over draft rows. Speculation itself is the shared runtime's.

use crate::bundle::{BundleError, Qwen4Bundle};
use crate::gpu_forward::{dense_ref, Qwen4LayerCapture};
use hipfire_dispatch::pipeline::{execute_embedding, EmbeddingOp};
use rdna_compute::{DType, Gpu, GpuTensor};
use smallvec::SmallVec;

/// The drafter's target layers and the device rows they are captured into.
pub(crate) struct Qwen4DflashTarget {
    layers: SmallVec<[usize; 8]>,
    /// F32 `[capture_rows × layers × hidden]`, sized by [`Qwen4Bundle::ensure_dflash_capture`].
    capture: Option<GpuTensor>,
    capture_rows: usize,
    /// One-row embedding lookup: token id (i32), F32 row, rotation staging.
    embed: Option<[GpuTensor; 3]>,
    /// Rows of the drafter's feature sink a prefill keeps (its window);
    /// `None` keeps every row.
    window: Option<usize>,
}

impl Qwen4DflashTarget {
    pub(crate) fn free_gpu(self, gpu: &mut Gpu) -> Result<(), hip_bridge::HipError> {
        let mut first = None;
        for tensor in self
            .capture
            .into_iter()
            .chain(self.embed.into_iter().flatten())
        {
            if let Err(error) = gpu.free_tensor(tensor) {
                first.get_or_insert(error);
            }
        }
        first.map_or(Ok(()), Err)
    }
}

fn forward_error(message: impl Into<String>) -> BundleError {
    BundleError::Forward(message.into())
}

impl Qwen4Bundle {
    /// Arm the DFlash2 capture of `layers` (zero-based, the drafter's
    /// `target_layer_ids`). Checked and allocated by
    /// [`Self::ensure_dflash_capture`]; an empty list disarms.
    pub(crate) fn set_dflash_layers(&mut self, layers: Vec<usize>) {
        self.dflash = (!layers.is_empty()).then(|| Qwen4DflashTarget {
            layers: layers.into_iter().collect(),
            capture: None,
            capture_rows: 0,
            embed: None,
            window: None,
        });
    }

    pub(crate) fn dflash_layers(&self) -> Option<&[usize]> {
        self.dflash.as_ref().map(|target| target.layers.as_slice())
    }

    /// The drafter reads only the last `rows` feature rows.
    pub(crate) fn set_dflash_window(&mut self, rows: usize) {
        if let Some(target) = self.dflash.as_mut() {
            target.window = Some(rows);
        }
    }

    /// Drop leading rows of the drafter's feature sink past its window.
    pub(crate) fn trim_dflash_rows(&self, out: &mut Vec<f32>) {
        if let Some(window) = self.dflash.as_ref().and_then(|target| target.window) {
            hipfire_runtime::dflash2_spec::trim_feature_rows(out, self.dflash_row_width(), window);
        }
    }

    /// Feature row width: `layers × hidden` F32.
    fn dflash_row_width(&self) -> usize {
        self.dflash
            .as_ref()
            .map_or(0, |target| target.layers.len() * self.config.hidden_size)
    }

    /// Capture rows for a `rows`-row verify and every prompt chunk, plus the
    /// embedding scratch; checks the armed layers against the trunk first.
    pub(crate) fn ensure_dflash_capture(
        &mut self,
        gpu: &mut Gpu,
        rows: usize,
    ) -> Result<(), BundleError> {
        self.ep_refuse("DFlash2 speculative decoding")?;
        if self.pp.is_some() || self.pp_band.is_some() {
            return Err(forward_error(
                "DFlash2 speculative decoding is not supported under qwen4 topology.pp",
            ));
        }
        let layers = self
            .dflash_layers()
            .ok_or_else(|| forward_error("DFlash2 target layers are not set"))?;
        let trunk = self.config.num_hidden_layers;
        // The PLE block opens its layer and adds to the streams before that
        // layer's mixer, so the layer before it has no clean capture point.
        let ple_layer = self
            .config
            .ple
            .as_ref()
            .and(self.config.ple_layer_ids.first())
            .and_then(|id| id.checked_sub(1));
        if layers.len() > 8
            || layers.windows(2).any(|pair| pair[0] >= pair[1])
            || layers.iter().any(|&layer| layer >= trunk)
            || layers.iter().any(|&layer| Some(layer + 1) == ple_layer)
        {
            return Err(forward_error(format!(
                "DFlash2 target layers {layers:?} must be at most 8 ascending trunk layers below \
                 {trunk}, none right before the PLE layer {ple_layer:?}"
            )));
        }
        // Per-row rollback points of a `rows`-row verify, so a partial accept
        // keeps its rows without a replay forward.
        let history = self.config.linear_conv_kernel_dim - 1;
        self.state
            .ensure_row_capture(gpu, rows, history)
            .map_err(|error| forward_error(error.to_string()))?;
        let chunk = self.spec_chunk_rows().unwrap_or(rows).max(1);
        let rows = crate::bundle::spec_hidden_capacity_rows(
            rows.max(chunk),
            chunk,
            self.npu_trunk_pipelined(),
        )
        .ok_or_else(|| forward_error("DFlash2 capture rows overflow"))?;
        let width = self.dflash_row_width();
        let hidden = self.config.hidden_size;
        let target = self.dflash.as_mut().expect("layers checked above");
        if target.capture_rows < rows {
            if let Some(old) = target.capture.take() {
                gpu.free_tensor(old).map_err(BundleError::Hip)?;
            }
            target.capture_rows = 0;
            let elements = rows
                .checked_mul(width)
                .ok_or_else(|| forward_error("DFlash2 capture rows overflow"))?;
            target.capture = Some(
                gpu.zeros(&[elements], DType::F32)
                    .map_err(BundleError::Hip)?,
            );
            target.capture_rows = rows;
        }
        if target.embed.is_none() {
            let ids = gpu.zeros(&[1], DType::F32).map_err(BundleError::Hip)?;
            let out = match gpu.zeros(&[hidden], DType::F32) {
                Ok(out) => out,
                Err(error) => {
                    let _ = gpu.free_tensor(ids);
                    return Err(BundleError::Hip(error));
                }
            };
            let rot = match gpu.zeros(&[hidden], DType::F32) {
                Ok(rot) => rot,
                Err(error) => {
                    let _ = gpu.free_tensor(ids);
                    let _ = gpu.free_tensor(out);
                    return Err(BundleError::Hip(error));
                }
            };
            target.embed = Some([ids, out, rot]);
        }
        Ok(())
    }

    /// The forward-side capture of a `rows`-row request when the DFlash2
    /// drafter is armed (`None` otherwise).
    pub(crate) fn dflash_layer_capture(
        &self,
        rows: usize,
    ) -> Result<Option<Qwen4LayerCapture>, BundleError> {
        let Some(target) = self.dflash.as_ref() else {
            return Ok(None);
        };
        let capture = target
            .capture
            .as_ref()
            .filter(|_| rows <= target.capture_rows)
            .ok_or_else(|| {
                forward_error(format!(
                    "DFlash2 capture holds {} rows, the forward needs {rows}",
                    target.capture_rows
                ))
            })?;
        Ok(Some(Qwen4LayerCapture {
            layers: target.layers.clone(),
            rows: capture.sub_offset(0, rows * self.dflash_row_width()),
            row: 0,
        }))
    }

    /// Append the first `rows` captured feature rows (`layers × hidden` F32
    /// each, layer-ascending) of the last capturing forward to `out`.
    pub(crate) fn read_dflash_capture(
        &self,
        gpu: &mut Gpu,
        rows: usize,
        out: &mut Vec<f32>,
    ) -> Result<(), BundleError> {
        let capture = self
            .dflash
            .as_ref()
            .and_then(|target| {
                target
                    .capture
                    .as_ref()
                    .filter(|_| rows <= target.capture_rows)
            })
            .ok_or_else(|| forward_error("DFlash2 capture is not allocated"))?;
        let elements = rows * self.dflash_row_width();
        let start = out.len();
        out.resize(start + elements, 0.0);
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(out[start..].as_mut_ptr() as *mut u8, elements * 4)
        };
        gpu.hip
            .memcpy_dtoh(bytes, &capture.sub_offset(0, elements).buf)
            .map_err(BundleError::Hip)?;
        gpu.take_device_wait_error().map_err(BundleError::Hip)
    }

    /// The embedding table's row of `token`, as F32 (the drafter's raw
    /// noise embedding: no scale, no PLE).
    pub(crate) fn dflash_embed_row(
        &mut self,
        gpu: &mut Gpu,
        token: u32,
    ) -> Result<Vec<f32>, BundleError> {
        let hidden = self.config.hidden_size;
        if token as usize >= self.config.vocab_size {
            return Err(forward_error(format!(
                "DFlash2 embedding token {token} is outside the vocabulary"
            )));
        }
        let [ids, out, rot] = self
            .dflash
            .as_ref()
            .and_then(|target| target.embed.as_ref())
            .ok_or_else(|| forward_error("DFlash2 embedding scratch is not allocated"))?;
        gpu.hip
            .memcpy_htod(&ids.buf, &(token as i32).to_ne_bytes())
            .map_err(BundleError::Hip)?;
        let table = self
            .weights
            .resident(&self.weights.root.embedding)
            .map_err(|error| forward_error(error.to_string()))?;
        execute_embedding(
            gpu,
            &EmbeddingOp {
                table,
                rotated: rot,
                token_ids: ids,
                output: out,
                rows: 1,
                dim: hidden,
            },
        )
        .map_err(|error| forward_error(format!("DFlash2 embedding: {error:?}")))?;
        gpu.download_f32(out).map_err(BundleError::Hip)
    }

    /// The target head over `rows` F32 rows of `hidden` (the draft's final
    /// rows), left on the device: a view of the spec logits, valid until the
    /// next speculative forward.
    pub(crate) fn dflash_lm_head(
        &mut self,
        gpu: &mut Gpu,
        hidden: &GpuTensor,
        rows: usize,
    ) -> Result<GpuTensor, BundleError> {
        let width = self.config.hidden_size;
        if rows == 0 || hidden.dtype != DType::F32 || hidden.numel() < rows * width {
            return Err(forward_error(format!(
                "DFlash2 head needs {rows} F32 rows of {width}"
            )));
        }
        let logits = self.spec_logits_rows(0, rows)?;
        let forward = self
            .execution
            .as_ref()
            .ok_or_else(|| forward_error("Qwen4 forward resources are not attached"))?;
        if forward.scratch.rotation.numel() < rows * width {
            return Err(forward_error("Qwen4 head rotation scratch is too small"));
        }
        let rotation = forward.scratch.rotation.sub_offset(0, rows * width);
        let lm_head = dense_ref(&self.weights, &self.weights.root.lm_head)
            .map_err(|error| forward_error(error.to_string()))?;
        // The draft rows take the gfx1151 PM row kernels like a verify forward.
        let rows_scope = std::mem::replace(&mut gpu.lane_rows_scope, true);
        let result = crate::program::execute_lm_head(
            gpu,
            &lm_head,
            &hidden.sub_offset(0, rows * width),
            &logits,
            rows,
            rows,
            Some(&rotation),
        );
        gpu.lane_rows_scope = rows_scope;
        result.map_err(|error| forward_error(format!("DFlash2 head: {error:?}")))?;
        Ok(logits)
    }
}
