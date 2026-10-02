// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Speculative-decode draft head: ranks the vocabulary for one hidden row and
//! returns the draft token.
//!
//! Drafts only steer acceptance; the target verifies every emitted token, so a
//! draft may rank a lower-bit copy of the language head.  With re-scoring the
//! copy's best 8 are re-scored exactly against the source head, which makes
//! the draft the source head's own argmax whenever the copy ranks it there.

use super::layer_ops::{execute_argmax, hip, project_weight};
use crate::families::gemv::WeightRef;
use crate::types::DispatchError;
use rdna_compute::{DType, Gpu, GpuTensor};

/// Ranking-copy policy: an optional lower-bit copy of the language head
/// that drafts rank the vocabulary with, and whether its top 8 are
/// re-scored exactly against the source head.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DraftHeadPolicy {
    pub copy: Option<DType>,
    pub rescore: bool,
}

impl DraftHeadPolicy {
    /// `mq2`..`mq6`, optional `r` suffix = rescore; anything else = no copy.
    pub fn parse(choice: &str) -> Self {
        let copy = match choice.trim_end_matches('r') {
            "mq6" => Some(DType::MQ6G256V2),
            "mq5" => Some(DType::MQ5G256V2),
            "mq4" => Some(DType::MQ4G256V2),
            "mq3" => Some(DType::MQ3G256V2),
            "mq2" => Some(DType::MQ2G256V2),
            _ => None,
        };
        Self {
            copy,
            rescore: choice.ends_with('r'),
        }
    }
}

/// Row layout drafts rank: `[0, front) ++ [special, vocab) ++ [front, special)`.
///
/// BPE merge order puts the frequent tokens first, so with re-scoring a draft
/// ranks only the first two ranges (`front` plus the control ids from
/// `special` on) while no recent input token fell in `[front, special)`.
#[derive(Clone, Copy, Debug)]
pub struct DraftHeadLayout {
    pub vocab: usize,
    pub hidden: usize,
    pub front: usize,
    pub special: usize,
    /// Drafts that rank the whole vocabulary after an input id in
    /// `[front, special)`.
    pub full_hold: u32,
}

pub struct DraftHead {
    /// Lower-bit ranking copy, rows in the layout order (`None` = rank the
    /// source head).
    copy: Option<GpuTensor>,
    /// Top-8 re-score scratch; present only when re-scoring.
    partial: Option<GpuTensor>,
    logits: GpuTensor,
    /// Draft token id, then (with re-scoring) its f32 logit margin.
    top1: GpuTensor,
    rotation: GpuTensor,
    /// 0 = whole vocabulary always, rows in token order.
    front: usize,
    special: usize,
    vocab: usize,
    hidden: usize,
    full_hold: u32,
    full_steps: u32,
    margin: f32,
}

/// Ranked row count and the `(front, special, tail)` order the re-score
/// kernel maps ranked rows back to token ids with.
fn ranking(
    front: usize,
    special: usize,
    vocab: usize,
    full_steps: u32,
) -> (usize, (usize, usize, usize)) {
    match (front, full_steps) {
        (0, _) => (vocab, (vocab, vocab, 0)),
        (front, 0) => (front + vocab - special, (front, special, vocab - special)),
        (front, _) => (vocab, (front, special, vocab - special)),
    }
}

/// `head` (`vocab` equal rows) reordered `[0, front) ++ [special, vocab)
/// ++ [front, special)`; `front == 0` keeps it.
fn front_first(
    gpu: &mut Gpu,
    head: GpuTensor,
    front: usize,
    special: usize,
    vocab: usize,
) -> hip_bridge::HipResult<GpuTensor> {
    if front == 0 {
        return Ok(head);
    }
    let stride = head.buf.size() / vocab;
    let out = match gpu.alloc_tensor(&head.shape, head.dtype) {
        Ok(out) => out,
        Err(error) => {
            let _ = gpu.free_tensor(head);
            return Err(error);
        }
    };
    let mut dst = 0;
    let mut copied = Ok(());
    for (start, end) in [(0, front), (special, vocab), (front, special)] {
        let bytes = (end - start) * stride;
        copied = copied
            .and_then(|_| gpu.memcpy_dtod_at_auto(&out.buf, dst, &head.buf, start * stride, bytes));
        dst += bytes;
    }
    let copied = copied.and_then(|_| gpu.hip.device_synchronize());
    // Back to the device: nothing else reuses a pooled vocab-sized copy.
    let freed = gpu.release_tensor_immediate(head);
    if let Err(error) = copied.and(freed) {
        let _ = gpu.free_tensor(out);
        return Err(error);
    }
    Ok(out)
}

/// What [`DraftHead::new`] builds for a `head_dtype` head: the ranking-copy
/// format, whether the copy's top 8 are re-scored, and the layout front.
fn plan(
    head_dtype: DType,
    layout: DraftHeadLayout,
    policy: DraftHeadPolicy,
) -> (Option<DType>, bool, usize) {
    // Only a bigger head is worth a smaller copy.
    let copy_format = policy.copy.filter(|&format| {
        matches!(
            head_dtype,
            DType::Q8_0 | DType::BF16 | DType::MQ6G256V2 | DType::MQ5G256V2
        ) && format != head_dtype
    });
    // The re-score kernel reads Q8_0 or MQ6G256V2 rows at K = 2560.
    let rescore = policy.rescore
        && copy_format.is_some()
        && matches!(head_dtype, DType::Q8_0 | DType::MQ6G256V2)
        && layout.hidden == 2560;
    let front = if rescore && layout.front < layout.special {
        layout.front
    } else {
        0
    };
    (copy_format, rescore, front)
}

impl DraftHead {
    /// `(resident, load scratch)` device bytes [`Self::new`] takes for a
    /// `head_dtype` head: what the head keeps, and the most it holds on top
    /// of that while building — the F32 requant scratch, or the unordered
    /// copy while rows are reordered. Both go back to the device, not the
    /// pool, before `new` returns.
    pub fn device_bytes(
        head_dtype: DType,
        layout: DraftHeadLayout,
        policy: DraftHeadPolicy,
    ) -> Option<(usize, usize)> {
        let (copy_format, rescore, front) = plan(head_dtype, layout, policy);
        let mut resident = layout
            .vocab
            .checked_mul(std::mem::size_of::<f32>())?
            .checked_add(8)?
            .checked_add(layout.hidden.checked_mul(std::mem::size_of::<f32>())?)?;
        let mut scratch = 0;
        if let Some(format) = copy_format {
            let (values, copy) = Gpu::requant_g256_bytes(layout.vocab, layout.hidden, format)?;
            resident = resident.checked_add(copy)?;
            scratch = if front == 0 { values } else { values.max(copy) };
        }
        if rescore {
            resident = resident.checked_add(Gpu::TOPK8_PARTIAL_BYTES)?;
        }
        Some((resident, scratch))
    }

    pub fn new(
        gpu: &mut Gpu,
        head: &GpuTensor,
        layout: DraftHeadLayout,
        policy: DraftHeadPolicy,
    ) -> Result<Self, DispatchError> {
        let (copy_format, rescore, front) = plan(head.dtype, layout, policy);
        let mut owned: Vec<GpuTensor> = Vec::with_capacity(5);
        let allocated = (|| -> hip_bridge::HipResult<()> {
            if let Some(format) = copy_format {
                let copy = gpu.requant_g256(head, layout.vocab, layout.hidden, format)?;
                owned.push(front_first(gpu, copy, front, layout.special, layout.vocab)?);
            }
            if rescore {
                owned.push(gpu.zeros(&[Gpu::TOPK8_PARTIAL_BYTES], DType::Raw)?);
            }
            owned.push(gpu.zeros(&[layout.vocab], DType::F32)?);
            owned.push(gpu.zeros(&[8], DType::Raw)?);
            owned.push(gpu.zeros(&[layout.hidden], DType::F32)?);
            Ok(())
        })();
        if let Err(error) = allocated {
            for tensor in owned {
                let _ = gpu.free_tensor(tensor);
            }
            return Err(DispatchError::Hip(error.to_string()));
        }
        let mut owned = owned.into_iter();
        let mut next = || owned.next().expect("draft head allocation count");
        let copy = copy_format.map(|_| next());
        let partial = rescore.then(&mut next);
        Ok(Self {
            copy,
            partial,
            logits: next(),
            top1: next(),
            rotation: next(),
            front,
            special: layout.special,
            vocab: layout.vocab,
            hidden: layout.hidden,
            full_hold: layout.full_hold,
            full_steps: 0,
            margin: f32::INFINITY,
        })
    }

    /// An input id in `[front, special)` ranks the whole vocabulary for
    /// `full_hold` drafts.
    pub fn observe(&mut self, token: u32) {
        if (self.front..self.special).contains(&(token as usize)) {
            self.full_steps = self.full_hold;
        }
    }

    /// Rank `hidden` (`[hidden]` F32) and return the draft token id.
    pub fn draft(
        &mut self,
        gpu: &mut Gpu,
        head: &GpuTensor,
        hidden: &GpuTensor,
    ) -> Result<u32, DispatchError> {
        let (ranked, order) = ranking(self.front, self.special, self.vocab, self.full_steps);
        self.full_steps = self.full_steps.saturating_sub(1);
        let ranking_head = self.copy.as_ref().unwrap_or(head);
        project_weight(
            gpu,
            &WeightRef {
                buf: ranking_head,
                dtype: ranking_head.dtype,
                m: ranked,
                k: self.hidden,
                row_stride: ranking_head
                    .dtype
                    .row_bytes(self.hidden)
                    .unwrap_or(self.hidden * ranking_head.dtype.size()),
                rotation: None,
                awq_scale: None,
                lloyd_lut_e4m3: None,
                lloyd_lut_f16: None,
                lloyd_lut_c16: None,
            },
            hidden,
            &self.logits,
            1,
            Some(&self.rotation),
        )?;
        if let Some(partial) = self.partial.as_ref() {
            // An MQ6 head reads the ranking copy's FWHT-rotated input.
            let x = if head.dtype == DType::Q8_0 {
                hidden
            } else {
                &self.rotation
            };
            hip(gpu.topk8_rescore_k2560(
                &self.logits,
                ranked,
                order,
                head,
                x,
                partial,
                &self.top1,
            ))?;
        } else {
            execute_argmax(gpu, &self.logits, &self.top1, 1, ranked)?;
        }
        let mut top = [0u8; 8];
        hip(gpu.hip.memcpy_dtoh(&mut top, &self.top1.buf))?;
        let token = u32::from_ne_bytes([top[0], top[1], top[2], top[3]]);
        self.margin = if self.partial.is_some() {
            f32::from_ne_bytes([top[4], top[5], top[6], top[7]])
        } else {
            f32::INFINITY
        };
        if token as usize >= self.vocab {
            return Err(DispatchError::Hip(format!(
                "draft token {token} is outside vocab {}",
                self.vocab
            )));
        }
        Ok(token)
    }

    /// Exact logit margin of the last draft over its runner-up among the
    /// re-scored candidates (infinite without re-scoring).
    pub fn margin(&self) -> f32 {
        self.margin
    }

    /// Logits of the last draft, rows in the ranking layout.
    pub fn logits(&self) -> &GpuTensor {
        &self.logits
    }

    pub fn free_gpu(self, gpu: &mut Gpu) -> Option<hip_bridge::HipError> {
        let mut first = None;
        let tensors = [
            Some(self.logits),
            Some(self.top1),
            Some(self.rotation),
            self.copy,
            self.partial,
        ];
        for tensor in tensors.into_iter().flatten() {
            if let Err(error) = gpu.free_tensor(tensor) {
                first.get_or_insert(error);
            }
        }
        first
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_parses_copy_tier_and_rescore_suffix() {
        let parse = DraftHeadPolicy::parse;
        assert_eq!(
            parse("mq2r"),
            DraftHeadPolicy {
                copy: Some(DType::MQ2G256V2),
                rescore: true
            }
        );
        assert_eq!(
            parse("mq6"),
            DraftHeadPolicy {
                copy: Some(DType::MQ6G256V2),
                rescore: false
            }
        );
        assert_eq!(
            parse("off"),
            DraftHeadPolicy {
                copy: None,
                rescore: false
            }
        );
        assert_eq!(
            parse("mq9r"),
            DraftHeadPolicy {
                copy: None,
                rescore: true
            }
        );
    }

    #[test]
    fn ranking_orders_front_special_tail_and_holds_full_vocab() {
        let (vocab, front, special) = (1000, 600, 900);
        assert_eq!(ranking(0, special, vocab, 3), (vocab, (vocab, vocab, 0)));
        assert_eq!(
            ranking(front, special, vocab, 0),
            (front + vocab - special, (front, special, vocab - special))
        );
        assert_eq!(
            ranking(front, special, vocab, 1),
            (vocab, (front, special, vocab - special))
        );
    }
}
