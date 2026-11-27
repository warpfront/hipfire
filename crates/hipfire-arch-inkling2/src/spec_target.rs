// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Scalar interleaved [`SpecTarget`] for the Inkling-2-Mini DSpark committed
//! binding (`MTP-PLAN.md` §2, slice C): a correctness bridge, not a speed path.
//!
//! The target forward is reached only through [`InklingTapForward`], the
//! scalar tap seam (`inkling/dspark-body/SEAM.md`): one committed row at a
//! time, returning its greedy pick and writing its DSpark tap row. Verify
//! walks the window causally: forward the anchor, and forward draft `i` only
//! while it equals the target's pick after the previous row (stopping after an
//! accepted EOS). A rejected draft is never forwarded, so the target consumes
//! exactly the committed rows and needs no rollback; [`SpecTarget::commit_prefix`]
//! only checks that the driver's accepted prefix is the walk's.

use crate::dspark_body::{TAP_ROW, TAP_VISITS};
use hipfire_runtime::spec::{SpecAdvance, SpecScratch, SpecTapSink, SpecTarget};
use rdna_compute::{DType, Gpu, GpuTensor};

/// The scalar tap seam on the target forward.
pub trait InklingTapForward {
    /// Forward `token` as the committed row at absolute `position` (the next
    /// row: `position` equals the rows already consumed). Writes the row's
    /// DSpark taps into `taps` (`[4, 2304]` F32: the streams entering visits
    /// 5, 23 and 41, then the final normalized state) and returns the greedy
    /// argmax of the row's logits.
    fn forward_row_taps(
        &mut self,
        gpu: &mut Gpu,
        token: u32,
        position: usize,
        taps: &GpuTensor,
    ) -> Result<u32, String>;
    /// Drop all sequence state (position 0).
    fn reset_state(&mut self, gpu: &mut Gpu) -> Result<(), String>;
    /// Return the target to its state before the last verify window, which
    /// started at `window_start`, then forward `replay` (the window's
    /// committable prefix) from there, discarding its taps. `Ok(false)`: no
    /// restorable window (the default: the scalar seam keeps no snapshot), and
    /// the target is unchanged.
    fn restore_window(
        &mut self,
        _gpu: &mut Gpu,
        _window_start: usize,
        _replay: &[u32],
    ) -> Result<bool, String> {
        Ok(false)
    }
    fn eos_token(&self) -> u32;
    fn ctx_capacity(&self) -> usize;
}

/// [`SpecTarget`] over an [`InklingTapForward`].
pub struct InklingSpecTarget<F> {
    fwd: F,
    /// Rows consumed so far (the next row's position).
    position: usize,
    eos: u32,
}

/// The walk of the outstanding verify: `(window start, rows consumed)`.
#[derive(Default)]
pub struct InklingSpecScratch {
    pending: Option<(usize, usize)>,
}

impl SpecScratch for InklingSpecScratch {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn free(self: Box<Self>, _gpu: &mut Gpu) {}

    fn clear(&mut self) {
        self.pending = None;
    }
}

fn scratch(s: &mut dyn SpecScratch) -> Result<&mut InklingSpecScratch, String> {
    s.as_any_mut()
        .downcast_mut::<InklingSpecScratch>()
        .ok_or_else(|| "inkling spec target: foreign verify scratch".to_string())
}

fn tap_view(t: &GpuTensor, row: usize) -> GpuTensor {
    GpuTensor {
        buf: t.buf.byte_view(row * TAP_ROW * 4, TAP_ROW * 4),
        shape: vec![TAP_ROW],
        dtype: DType::F32,
    }
}

impl<F: InklingTapForward> InklingSpecTarget<F> {
    pub fn new(fwd: F) -> Self {
        let eos = fwd.eos_token();
        Self {
            fwd,
            position: 0,
            eos,
        }
    }

    pub fn forward(&mut self) -> &mut F {
        &mut self.fwd
    }

    pub fn into_inner(self) -> F {
        self.fwd
    }

    /// Rows consumed so far.
    pub fn position(&self) -> usize {
        self.position
    }

    fn check_position(&self, position: usize) -> Result<(), String> {
        if position != self.position {
            return Err(format!(
                "inkling spec target: row at {position} but {} rows are consumed",
                self.position
            ));
        }
        Ok(())
    }

    /// The causal walk: forward `block[0]`, then `block[i]` while it equals the
    /// pick after row `i - 1`, stopping after an accepted EOS. Row `i`'s taps go
    /// to `capture` row `i`. Picks past the walk are `u32::MAX` (never read by
    /// the greedy acceptance, which stops at the first mismatch).
    fn walk(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        position: usize,
        s: &mut dyn SpecScratch,
        capture: &GpuTensor,
    ) -> Result<Vec<u32>, String> {
        self.check_position(position)?;
        let sc = scratch(s)?;
        if sc.pending.is_some() {
            return Err("inkling spec target: verify while another is uncommitted".into());
        }
        if block.is_empty() || capture.buf.size() < block.len() * TAP_ROW * 4 {
            return Err("inkling spec target: verify capture too small".into());
        }
        let mut picks = vec![u32::MAX; block.len()];
        let mut consumed = 0;
        for i in 0..block.len() {
            let pick =
                self.fwd
                    .forward_row_taps(gpu, block[i], position + i, &tap_view(capture, i))?;
            picks[i] = pick;
            consumed = i + 1;
            self.position = position + consumed;
            let accepted_eos = i > 0 && block[i] == self.eos;
            if accepted_eos || i + 1 == block.len() || block[i + 1] != pick {
                break;
            }
        }
        sc.pending = Some((position, consumed));
        Ok(picks)
    }
}

impl<F: InklingTapForward + 'static> SpecTarget for InklingSpecTarget<F> {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn reset_recurrent(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        self.fwd.reset_state(gpu)?;
        self.position = 0;
        Ok(())
    }

    fn new_spec_scratch(
        &mut self,
        _gpu: &mut Gpu,
        _block_size: usize,
    ) -> Result<Box<dyn SpecScratch>, String> {
        Ok(Box::new(InklingSpecScratch::default()))
    }

    fn spec_advance(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        start_pos: usize,
        reset: bool,
        abort: &dyn Fn() -> bool,
        hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<SpecAdvance, String> {
        if reset {
            self.reset_recurrent(gpu)?;
        }
        self.check_position(start_pos)?;
        if tokens.is_empty() {
            return Err("inkling spec target: empty advance".into());
        }
        let row = gpu
            .alloc_tensor(&[TAP_ROW], DType::F32)
            .map_err(|e| format!("inkling spec target: alloc tap row: {e:?}"))?;
        let mut out = hidden_out;
        let mut last = 0;
        let mut result = Ok(());
        for (i, &t) in tokens.iter().enumerate() {
            if abort() {
                let _ = gpu.free_tensor(row);
                self.reset_recurrent(gpu)?;
                return Ok(SpecAdvance::Aborted);
            }
            match self.fwd.forward_row_taps(gpu, t, start_pos + i, &row) {
                Ok(p) => last = p,
                Err(e) => {
                    result = Err(e);
                    break;
                }
            }
            self.position = start_pos + i + 1;
            if let Some(sink) = out.as_deref_mut() {
                match gpu.download_f32(&row) {
                    Ok(v) => sink.extend_from_slice(&v),
                    Err(e) => {
                        result = Err(format!("inkling spec target: tap d2h: {e:?}"));
                        break;
                    }
                }
            }
        }
        let _ = gpu.free_tensor(row);
        result?;
        Ok(SpecAdvance::Ready {
            last_argmax: last,
            last_logits: None,
        })
    }

    fn spec_advance_taps_gpu(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        start_pos: usize,
        abort: &dyn Fn() -> bool,
        sink: &mut dyn SpecTapSink,
    ) -> Result<SpecAdvance, String> {
        self.check_position(start_pos)?;
        if tokens.is_empty() {
            return Err("inkling spec target: empty advance".into());
        }
        let cap = sink.rows_capacity();
        if cap == 0 || sink.chunk().buf.size() < cap * TAP_ROW * 4 {
            return Err("inkling spec target: tap chunk smaller than its row capacity".into());
        }
        let mut last = 0;
        let mut done = 0;
        while done < tokens.len() {
            let n = (tokens.len() - done).min(cap);
            for i in 0..n {
                if abort() {
                    // Rows already forwarded in this chunk are delivered, then
                    // the state resets: the caller drops both.
                    if i > 0 {
                        sink.on_rows(gpu, start_pos + done, i)?;
                    }
                    self.reset_recurrent(gpu)?;
                    return Ok(SpecAdvance::Aborted);
                }
                let pos = start_pos + done + i;
                let view = tap_view(sink.chunk(), i);
                last = self
                    .fwd
                    .forward_row_taps(gpu, tokens[done + i], pos, &view)?;
                self.position = pos + 1;
            }
            sink.on_rows(gpu, start_pos + done, n)?;
            done += n;
        }
        Ok(SpecAdvance::Ready {
            last_argmax: last,
            last_logits: None,
        })
    }

    fn verify_block(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        position: usize,
        s: &mut dyn SpecScratch,
        hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<Vec<u32>, String> {
        let capture = gpu
            .alloc_tensor(&[block.len().max(1) * TAP_ROW], DType::F32)
            .map_err(|e| format!("inkling spec target: alloc capture: {e:?}"))?;
        let walked = self.walk(gpu, block, position, s, &capture);
        let result = walked.and_then(|picks| {
            if let Some(sink) = hidden_out {
                let consumed = scratch(s)?.pending.map_or(0, |(_, n)| n);
                let v = gpu
                    .download_f32(&capture)
                    .map_err(|e| format!("inkling spec target: tap d2h: {e:?}"))?;
                sink.extend_from_slice(&v[..consumed * TAP_ROW]);
            }
            Ok(picks)
        });
        let _ = gpu.free_tensor(capture);
        result
    }

    fn verify_block_capture_gpu(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        position: usize,
        s: &mut dyn SpecScratch,
        hidden_gpu: &GpuTensor,
    ) -> Result<(Vec<u32>, bool), String> {
        // Rows `0..consumed` are written: the anchor plus every accepted draft,
        // the only rows the committed binding reads.
        Ok((self.walk(gpu, block, position, s, hidden_gpu)?, true))
    }

    fn commit_prefix(
        &mut self,
        _gpu: &mut Gpu,
        _block: &[u32],
        accept_len: usize,
        position: usize,
        s: &mut dyn SpecScratch,
    ) -> Result<(), String> {
        let sc = scratch(s)?;
        let Some((start, consumed)) = sc.pending.take() else {
            return Err("inkling spec target: commit without a verify".into());
        };
        if start != position || consumed != accept_len + 1 {
            return Err(format!(
                "inkling spec target: the walk consumed {consumed} rows at {start}, commit keeps {} at {position}",
                accept_len + 1
            ));
        }
        Ok(())
    }

    fn abandon_verify(&mut self, _gpu: &mut Gpu, s: &mut dyn SpecScratch) -> Result<(), String> {
        // The walk's rows are committed target state; without a snapshot of
        // the 48 visit slots (Kaden's seam, SEAM.md) they cannot be undone.
        match scratch(s)?.pending.take() {
            None => Ok(()),
            Some((start, consumed)) => Err(format!(
                "inkling spec target: cannot roll back {consumed} walked rows at {start}"
            )),
        }
    }

    fn repair_committed_prefix(
        &mut self,
        gpu: &mut Gpu,
        s: &mut dyn SpecScratch,
        position: usize,
        replay: &[u32],
    ) -> Result<bool, String> {
        if scratch(s)?.pending.is_some() {
            return Err("inkling spec target: repair while a verify is uncommitted".into());
        }
        if position + replay.len() > self.position {
            return Ok(false);
        }
        if !self.fwd.restore_window(gpu, position, replay)? {
            return Ok(false);
        }
        self.position = position + replay.len();
        Ok(true)
    }

    fn eos_token(&self) -> u32 {
        self.eos
    }

    fn ctx_capacity(&self) -> usize {
        self.fwd.ctx_capacity()
    }

    fn dflash_extract_layers(&self) -> Option<&[usize]> {
        Some(&TAP_VISITS)
    }
}
