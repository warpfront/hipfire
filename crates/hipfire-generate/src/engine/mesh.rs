// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! [`Forwards`] over any [`MeshFamily`] on any mesh: every forward is a
//! mesh epoch the shared [`MeshRunner`] submits and drains.

use super::{Forwards, Next};
use hipfire_runtime::mesh_family::{DecodeOut, MeshFamily, MeshRequest, MeshRunner};
use hipfire_runtime::multi_gpu::Gpus;
use std::ops::Range;
use std::time::Instant;

pub struct MeshForwards<'a> {
    pub runner: &'a mut MeshRunner,
    pub request: &'a MeshRequest,
    pub family: &'a mut dyn MeshFamily,
    pub gpus: &'a mut Gpus,
    pub prompt: &'a [u32],
    /// Decode rows the request may still write (mapped ahead after the prompt).
    pub budget: usize,
    /// When the prompt's final epoch drained (the prefill/decode split).
    pub prompt_done: Option<Instant>,
}

impl Forwards for MeshForwards<'_> {
    fn prefill(&mut self, rows: Range<usize>) -> Result<(usize, Option<Vec<f32>>), String> {
        if self.family.position() != rows.start {
            return Err(format!(
                "prompt epoch at {} but the model is at {}",
                rows.start,
                self.family.position()
            ));
        }
        let (end, logits) =
            self.runner
                .prompt_epoch(self.request, &mut *self.family, self.gpus, self.prompt)?;
        if logits.is_some() {
            self.prompt_done = Some(Instant::now());
            self.family.reserve_decode(self.gpus, self.budget);
        }
        Ok((end, logits))
    }

    fn decode(&mut self, token: Option<u32>, top1: bool) -> Result<(u32, Next), String> {
        let (ran, out) = self.runner.decode_epoch(
            self.request,
            &mut *self.family,
            self.gpus,
            token,
            top1,
            true,
        )?;
        let next = match out {
            DecodeOut::Top1(id) => Next::Top1(id),
            DecodeOut::Logits(row) => Next::Logits(row),
            DecodeOut::Device => Next::Device,
        };
        Ok((ran, next))
    }

    fn prefill_rerun(&mut self) -> Result<Option<usize>, String> {
        let Some(start) = self.family.take_prompt_rerun(self.prompt) else {
            return Ok(None);
        };
        self.runner.recover(&*self.family, self.gpus)?;
        self.family.begin_prompt(self.gpus, self.prompt, start)?;
        Ok(Some(start))
    }
}
