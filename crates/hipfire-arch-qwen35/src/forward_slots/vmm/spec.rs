// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Speculative (MTP) lanes of the VMM continuous-batching store.
//!
//! A spec lane is a request decoded exactly like the singleton MTP route
//! (`Qwen35MtpDrafter`): its prompt is filled by the singleton MTP prompt
//! prefill on the request's own KV/DeltaNet, then every decode window is a
//! `RequestStepKind::Verify` request of an executor step — per-request draft
//! (provision), the verify trunk and head shared with every other Verify
//! request (forward), per-request accept/repair (commit). Per request
//! the committed ids and the full state (trunk KV, DeltaNet, MTP head KV,
//! `prev_hidden`) follow its isolated singleton spec run. Sampled lanes keep
//! the singleton's request-local MtpRng, generation-only history and window
//! budget; sampled singleton promotion is not supported by `spec_adopt`.

use super::{vmm_map_room, Qwen35RequestState, Qwen35VmmStore};
use crate::mtp_head::{MtpKvMode, Qwen35MtpHead};
use crate::mtp_spec::cb::{
    mtp_cb_accept, mtp_cb_draft_batched, mtp_cb_verify, MtpCbDraftLane, MtpCbScratch,
    MtpCbVerified, MtpCbVerifyLane,
};
use crate::mtp_spec::MtpDraftOutput;
use crate::mtp_spec::{
    prefill_trunk_and_mtp_cache_parts, sample_from_logits, MtpPrefillTarget, MtpPromptRoute,
};
use crate::mtp_speculator::{new_qwen35_mtp_lane_state, Qwen35MtpDrafter};
use crate::qwen35::{DeltaNetState, Qwen35Config, Qwen35Scratch, Qwen35Weights};
use hipfire_runtime::slot_batch::{
    BatchStepPlan, RequestAdvance, RequestEpoch, RequestRows, RequestStepKind,
};
use hipfire_runtime::spec::{SpecRequestConfig, Speculator};
use rdna_compute::{DType, Gpu, GpuTensor};

/// The store's MTP engine: its own head copy (KV sized `head_cap`), the
/// window size `k`, the shared verify scratch, and the probed device bytes of
/// one lane's MTP state (charged by memory admission).
pub struct VmmSpecEngine {
    head: Qwen35MtpHead,
    k: usize,
    cb: MtpCbScratch,
    route: MtpPromptRoute,
    lane_bytes: usize,
}

impl VmmSpecEngine {
    /// `head` loaded with the per-lane context cap; `k` = the singleton
    /// speculator's window (`mtp_k`, clamped like `Qwen35MtpDrafter`);
    /// `max_lanes * (k + 1)` plans the shared verify scratch rows, clamped to
    /// the effective chunk cap (63, or up to 128 on the wide verify route, default on for
    /// gfx1201, `HIPFIRE_CB_VERIFY_CHUNK128=0` restores 63; [`MtpCbScratch::new`]).
    ///
    /// One lane's MTP state is probed for the store's memory admission:
    /// allocated shaped from `template_dn` exactly as a lane is
    /// ([`new_qwen35_mtp_lane_state`], greedy default request, penalty window
    /// bound `ar_cap` = the AR `repeat_buf` capacity in ids), the live-byte
    /// drop (HIP free plus pool-parked) is measured, then it is freed (the
    /// buffers park in the pool, where the first lane reuses them).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gpu: &mut Gpu,
        config: &Qwen35Config,
        template_dn: &DeltaNetState,
        head: Qwen35MtpHead,
        k: usize,
        max_lanes: usize,
        route: MtpPromptRoute,
        ar_cap: usize,
    ) -> Result<Self, String> {
        let k = k.clamp(1, 8);
        let cb = match MtpCbScratch::new(gpu, config, max_lanes * (k + 1)) {
            Ok(cb) => cb,
            Err(e) => {
                head.free_gpu(gpu);
                return Err(format!("VMM spec engine: verify scratch: {e}"));
            }
        };
        let live = |gpu: &Gpu| {
            gpu.device_mem_info()
                .map(|(f, _)| f + gpu.pool_parked_bytes())
        };
        let probed = (|| -> Result<usize, String> {
            let before = live(gpu).map_err(|e| e.to_string())?;
            let st = new_qwen35_mtp_lane_state(
                gpu,
                config,
                template_dn,
                &head,
                k,
                SpecRequestConfig::default(),
                ar_cap,
            )?;
            let after = live(gpu).map_err(|e| e.to_string());
            st.free_gpu(gpu);
            Ok(before.saturating_sub(after?))
        })();
        match probed {
            Ok(lane_bytes) => Ok(Self {
                head,
                k,
                cb,
                route,
                lane_bytes,
            }),
            Err(e) => {
                let _ = cb.free_gpu(gpu);
                head.free_gpu(gpu);
                Err(format!("VMM spec engine: lane footprint probe: {e}"))
            }
        }
    }

    /// Probed device bytes of one lane's MTP state.
    pub fn lane_bytes(&self) -> usize {
        self.lane_bytes
    }

    pub fn k(&self) -> usize {
        self.k
    }

    /// Exclusive position bound of a spec lane (MTP head KV rows).
    pub fn head_cap(&self) -> usize {
        self.head.config.max_seq
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = self.cb.free_gpu(gpu);
        self.head.free_gpu(gpu);
    }
}

impl Qwen35VmmStore {
    /// Install the MTP engine (staging; once). Spec lanes need it.
    pub fn install_spec(&mut self, engine: VmmSpecEngine) -> Result<(), VmmSpecEngine> {
        if self.spec.is_some() {
            return Err(engine);
        }
        self.spec = Some(engine);
        Ok(())
    }

    pub fn spec_engine(&self) -> Option<&VmmSpecEngine> {
        self.spec.as_ref()
    }

    /// Map KV positions `[0, end)` of `epoch` against the shared physical
    /// budget and the device memory a map can use (as `provision_step`).
    pub(super) fn spec_provision(
        &mut self,
        gpu: &mut Gpu,
        epoch: &RequestEpoch,
        end: usize,
    ) -> Result<(), String> {
        let mapped_total = self.mapped_kv_bytes()?;
        let device_room = vmm_map_room(
            gpu.device_mem_info()
                .map_err(|e| format!("spec provision: VRAM query: {e}"))?
                .0,
            gpu.pool_parked_bytes(),
        );
        let budget = self
            .kv_budget_bytes
            .saturating_sub(mapped_total)
            .min(device_room);
        let s = self
            .slots
            .iter_mut()
            .flatten()
            .find(|s| s.epoch == *epoch)
            .ok_or_else(|| format!("spec provision: unknown epoch {epoch:?}"))?;
        if end > s.kv.vmm_logical_bound() {
            return Err(format!(
                "spec provision: {end} positions > logical bound {}",
                s.kv.vmm_logical_bound()
            ));
        }
        let grown =
            s.kv.provision_vmm_positions(gpu, end, budget)
                .map_err(|e| format!("spec provision: map {end} positions for {epoch:?}: {e}"))?;
        self.mapped_high_water = self.mapped_high_water.max(mapped_total + grown);
        Ok(())
    }

    /// Make `epoch` a spec lane and fill its prompt by the singleton MTP
    /// prompt prefill (cold: fresh owners, positions from 0), its MTP state
    /// configured with `request` as the singleton speculator would be.
    /// Returns the first generated id (greedy: host argmax; sampled: the
    /// singleton's unpenalized host sample) and leaves it as the lane's pending
    /// seed. On error the lane is poisoned.
    #[allow(clippy::too_many_arguments)]
    pub fn spec_prefill(
        &mut self,
        gpu: &mut Gpu,
        weights: &Qwen35Weights,
        config: &Qwen35Config,
        scratch: &Qwen35Scratch,
        epoch: &RequestEpoch,
        prompt: &[u32],
        request: SpecRequestConfig,
    ) -> Result<u32, String> {
        if !matches!(self.phase, super::Phase::Idle) {
            return Err("spec prefill during an uncommitted step".into());
        }
        let engine = self
            .spec
            .as_ref()
            .ok_or("spec prefill: no MTP engine staged")?;
        if prompt.is_empty() || prompt.len() >= engine.head_cap() {
            return Err(format!(
                "spec prefill: prompt {} outside 1..{}",
                prompt.len(),
                engine.head_cap()
            ));
        }
        let (route, k) = (engine.route, engine.k);
        self.spec_provision(gpu, epoch, prompt.len())?;
        let Self { slots, spec, .. } = self;
        let engine = spec.as_ref().expect("checked above");
        let s = slots
            .iter_mut()
            .flatten()
            .find(|s| s.epoch == *epoch)
            .ok_or_else(|| format!("spec prefill: unknown epoch {epoch:?}"))?;
        let r = (|| -> Result<u32, String> {
            if s.position != 0 || s.mtp.is_some() || s.dflash.is_some() {
                return Err("spec prefill: lane is not fresh".into());
            }
            // AR's penalty window bound: this scratch's `repeat_buf` (the
            // singleton drafter reads its slot's).
            let ar_cap = scratch.repeat_buf.buf.size() / 4;
            let mut st =
                new_qwen35_mtp_lane_state(gpu, config, &s.dn, &engine.head, k, request, ar_cap)
                    .map_err(|e| format!("spec prefill: MTP state: {e}"))?;
            let filled = st.reset(gpu).map_err(|e| e.to_string()).and_then(|()| {
                let mut target = MtpPrefillTarget {
                    weights,
                    config,
                    kv_cache: &mut s.kv,
                    dn_state: &mut s.dn,
                    scratch,
                };
                prefill_trunk_and_mtp_cache_parts(
                    gpu,
                    &mut target,
                    &engine.head,
                    &mut st,
                    prompt,
                    0,
                    route,
                    |_, _, _, _| Ok(()),
                )
                .map_err(|e| e.to_string())
            });
            s.mtp = Some(st);
            filled.map_err(|e| format!("spec prefill: {e}"))?;
            // Mirror Qwen35MtpDrafter::mtp_prefill (mtp_speculator.rs:487-501):
            // the first token is unpenalized, and consumes the SAME MtpRng
            // subsequently used for drafts, acceptance, residual and bonus.
            let logits = gpu
                .download_f32(&scratch.logits)
                .map_err(|e| format!("spec prefill: seed logits: {e}"))?;
            let st = s.mtp.as_mut().expect("installed above");
            let seed = if st.sampling.is_greedy() {
                logits
                    .iter()
                    .enumerate()
                    .fold((0u32, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                        if v > bv {
                            (i as u32, v)
                        } else {
                            (bi, bv)
                        }
                    })
                    .0
            } else {
                sample_from_logits(&logits, &st.sampling, &mut st.rng).0
            };
            s.position = prompt.len();
            s.pending_seed = Some(seed);
            s.history.push(seed);
            Ok(seed)
        })();
        if r.is_err() {
            s.poisoned = true;
        }
        r
    }

    /// Continue a promoted singleton MTP request as a spec lane: its admitted
    /// owners (trunk KV/DeltaNet, `position`, `pending_seed`) came from the
    /// singleton, `snap` carries its drafter state. The lane's MTP state is
    /// configured with `request` as on the singleton route and gets the
    /// singleton's head KV rows `[0, position)`, `prev_hidden` and
    /// `prev_hidden_pos`. Consumes `snap` either way.
    #[allow(clippy::too_many_arguments)]
    pub fn spec_adopt(
        &mut self,
        gpu: &mut Gpu,
        config: &Qwen35Config,
        scratch: &Qwen35Scratch,
        epoch: &RequestEpoch,
        snap: MtpLaneSnapshot,
        request: SpecRequestConfig,
    ) -> Result<(), String> {
        let r = self.spec_adopt_inner(gpu, config, scratch, epoch, &snap, request);
        snap.free_gpu(gpu);
        r
    }

    fn spec_adopt_inner(
        &mut self,
        gpu: &mut Gpu,
        config: &Qwen35Config,
        scratch: &Qwen35Scratch,
        epoch: &RequestEpoch,
        snap: &MtpLaneSnapshot,
        request: SpecRequestConfig,
    ) -> Result<(), String> {
        if !matches!(self.phase, super::Phase::Idle) {
            return Err("spec adopt during an uncommitted step".into());
        }
        let Self { slots, spec, .. } = self;
        let engine = spec.as_ref().ok_or("spec adopt: no MTP engine staged")?;
        let s = slots
            .iter_mut()
            .flatten()
            .find(|s| s.epoch == *epoch)
            .ok_or_else(|| format!("spec adopt: unknown epoch {epoch:?}"))?;
        if s.mtp.is_some()
            || s.dflash.is_some()
            || s.pending_seed.is_none()
            || s.position != snap.rows
        {
            return Err(format!(
                "spec adopt: lane position {} / snapshot rows {} / seed {:?}",
                s.position, snap.rows, s.pending_seed
            ));
        }
        if snap.rows >= engine.head_cap() {
            return Err(format!(
                "spec adopt: position {} >= MTP head capacity {}",
                snap.rows,
                engine.head_cap()
            ));
        }
        let ar_cap = scratch.repeat_buf.buf.size() / 4;
        let mut st =
            new_qwen35_mtp_lane_state(gpu, config, &s.dn, &engine.head, engine.k, request, ar_cap)
                .map_err(|e| format!("spec adopt: MTP state: {e}"))?;
        let installed = (|| -> Result<(), String> {
            st.reset(gpu).map_err(|e| e.to_string())?;
            let rb = snap.row_bytes;
            if st.mtp_kv.n_head_kv * (st.mtp_kv.head_dim / 32) * 34 != rb
                || st.prev_hidden.byte_size() != snap.prev_hidden.byte_size()
            {
                return Err("snapshot layout differs from the lane's MTP state".into());
            }
            let bytes = snap.rows * rb;
            for (dst, src) in [
                (&st.mtp_kv.inner.k_gpu[0], &snap.k_rows),
                (&st.mtp_kv.inner.v_gpu[0], &snap.v_rows),
            ] {
                gpu.memcpy_dtod_at_auto(&dst.buf, 0, &src.buf, 0, bytes)
                    .map_err(|e| e.to_string())?;
            }
            gpu.memcpy_dtod_at_auto(
                &st.prev_hidden.buf,
                0,
                &snap.prev_hidden.buf,
                0,
                snap.prev_hidden.byte_size(),
            )
            .map_err(|e| e.to_string())?;
            st.prev_hidden_pos = snap.prev_hidden_pos;
            gpu.hip.device_synchronize().map_err(|e| e.to_string())
        })();
        match installed {
            Ok(()) => {
                s.mtp = Some(st);
                Ok(())
            }
            Err(e) => {
                st.free_gpu(gpu);
                Err(format!("spec adopt: {e}"))
            }
        }
    }

    // ── RequestStepKind::Verify rows through provision → forward → commit ──
    //
    // A Verify request is tagged by its lane state: `mtp` (the MTP engine) or
    // `dflash` (the DFlash engine). Each tag's lanes run their own draft,
    // shared trunk + head and accept consumer; a step may carry both tags,
    // each tag's whole lanes packing into its own trunk chunks of at most the
    // effective cap (`<= 63` rows, or up to 128 on the wide verify route).

    fn is_mtp_lane(&self, epoch: &RequestEpoch) -> bool {
        self.request_state(epoch).is_some_and(|s| s.mtp.is_some())
    }

    /// Provision gate of one planned Verify request: a live spec lane (MTP or
    /// DFlash), its seed in the first row, rows within the lane's window.
    pub(super) fn spec_check_verify(
        &self,
        r: &RequestRows,
        draft_len: usize,
        seed_row_token: u32,
    ) -> Result<(), String> {
        let s = self
            .request_state(&r.epoch)
            .ok_or_else(|| format!("provision_step: stale or unknown epoch {:?}", r.epoch))?;
        if s.dflash.is_some() {
            return self.dflash_check_verify(s, r, draft_len, seed_row_token);
        }
        let engine = self
            .spec
            .as_ref()
            .ok_or("provision_step: Verify rows need the staged MTP engine")?;
        if s.mtp.is_none() || s.spec_draft.is_some() || s.spec_verified.is_some() {
            return Err(format!(
                "provision_step: {:?} is not an idle spec lane",
                r.epoch
            ));
        }
        if draft_len > engine.k || r.rows.len != draft_len + 1 {
            return Err(format!(
                "provision_step: Verify draft_len {draft_len} (rows {}) outside the MTP window {}",
                r.rows.len, engine.k
            ));
        }
        if s.pending_seed != Some(seed_row_token) {
            return Err(format!(
                "provision_step: Verify seed row {seed_row_token} != pending seed {:?}",
                s.pending_seed
            ));
        }
        if s.position + r.rows.len > engine.head_cap() {
            return Err(format!(
                "provision_step: Verify window end > MTP head capacity {}",
                engine.head_cap()
            ));
        }
        Ok(())
    }

    /// Provision: draft every planned Verify request (epoch-tagged: held in
    /// its own state until commit/abort). The verify rows the executor runs
    /// are `[seed, candidates…]` of this draft; planned rows past them are
    /// host-only placeholders. A draft touches only uncommitted drafter state
    /// (MTP head rows / DFlash draft scratch), so a failure drops every draft
    /// of the plan and maps nothing else.
    pub(super) fn spec_draft_planned(
        &mut self,
        gpu: &mut Gpu,
        weights: &Qwen35Weights,
        config: &Qwen35Config,
        plan: &BatchStepPlan,
    ) -> Result<(), String> {
        if !plan
            .requests
            .iter()
            .any(|r| matches!(r.kind, RequestStepKind::Verify { .. }))
        {
            return Ok(());
        }
        let r = self
            .dflash_draft_planned(gpu, weights, config, plan)
            .and_then(|()| self.mtp_draft_planned(gpu, weights, config, plan));
        if r.is_err() {
            self.spec_clear_planned(plan);
        }
        r
    }

    fn mtp_draft_planned(
        &mut self,
        gpu: &mut Gpu,
        weights: &Qwen35Weights,
        config: &Qwen35Config,
        plan: &BatchStepPlan,
    ) -> Result<(), String> {
        let planned: Vec<(RequestEpoch, usize)> = plan
            .requests
            .iter()
            .filter_map(|r| match r.kind {
                RequestStepKind::Verify { draft_len } if self.is_mtp_lane(&r.epoch) => {
                    Some((r.epoch, draft_len))
                }
                _ => None,
            })
            .collect();
        if planned.is_empty() {
            return Ok(());
        }
        let Self { slots, spec, .. } = self;
        let engine = spec.as_ref().expect("checked at provision");
        let mut owners: Vec<Option<&mut Qwen35RequestState>> =
            planned.iter().map(|_| None).collect();
        for s in slots.iter_mut().flatten() {
            if let Some(i) = planned.iter().position(|(e, _)| *e == s.epoch) {
                owners[i] = Some(s);
            }
        }
        let mut lanes = Vec::with_capacity(planned.len());
        let mut outs: Vec<&mut Option<MtpDraftOutput>> = Vec::with_capacity(planned.len());
        for (o, &(_, draft_len)) in owners.into_iter().zip(&planned) {
            let Qwen35RequestState {
                mtp,
                history,
                position,
                pending_seed,
                spec_draft,
                ..
            } = o.expect("checked at provision");
            // Mirror mtp_speculator.rs:545,566-572: request-budget k and
            // generated-only emitted history, already including the pending
            // seed once. mtp_cb_draft rebuilds the penalty window from this.
            lanes.push(MtpCbDraftLane {
                state: mtp.as_mut().expect("checked at provision"),
                cur_pos: *position,
                last_committed: pending_seed.expect("checked at provision"),
                emitted: history,
                k: draft_len,
            });
            outs.push(spec_draft);
        }
        match mtp_cb_draft_batched(gpu, weights, config, &engine.head, &engine.cb, &mut lanes) {
            Ok(drafts) => {
                for (slot, d) in outs.into_iter().zip(drafts) {
                    *slot = Some(d);
                }
                Ok(())
            }
            Err(e) => Err(format!("provision_step: draft: {e}")),
        }
    }

    /// Forward: verify every planned Verify request — each tag in its own
    /// shared trunk + head pass ([`mtp_cb_verify`] / `dflash_cb_verify`);
    /// outcomes wait in each state for commit.
    pub(super) fn spec_verify_planned(
        &mut self,
        gpu: &mut Gpu,
        weights: &Qwen35Weights,
        config: &Qwen35Config,
        scratch: &Qwen35Scratch,
        plan: &BatchStepPlan,
    ) -> hip_bridge::HipResult<()> {
        self.dflash_verify_planned(gpu, weights, config, scratch, plan)?;
        self.mtp_verify_planned(gpu, weights, config, scratch, plan)
    }

    fn mtp_verify_planned(
        &mut self,
        gpu: &mut Gpu,
        weights: &Qwen35Weights,
        config: &Qwen35Config,
        scratch: &Qwen35Scratch,
        plan: &BatchStepPlan,
    ) -> hip_bridge::HipResult<()> {
        let epochs: Vec<RequestEpoch> = plan
            .requests
            .iter()
            .filter(|r| {
                matches!(r.kind, RequestStepKind::Verify { .. }) && self.is_mtp_lane(&r.epoch)
            })
            .map(|r| r.epoch)
            .collect();
        if epochs.is_empty() {
            return Ok(());
        }
        let eos = config.eos_token;
        let Self { slots, spec, .. } = self;
        let engine = spec.as_ref().expect("checked at provision");
        let mut owners: Vec<Option<&mut Qwen35RequestState>> =
            epochs.iter().map(|_| None).collect();
        for s in slots.iter_mut().flatten() {
            if let Some(i) = epochs.iter().position(|e| *e == s.epoch) {
                owners[i] = Some(s);
            }
        }
        let mut outcomes_into: Vec<&mut Option<MtpCbVerified>> = Vec::with_capacity(epochs.len());
        let mut lanes: Vec<MtpCbVerifyLane<'_>> = Vec::with_capacity(epochs.len());
        for s in owners.into_iter() {
            let s =
                s.ok_or_else(|| hip_bridge::HipError::new(0, "verify: planned owner vanished"))?;
            let Qwen35RequestState {
                kv,
                dn,
                mtp,
                spec_draft,
                spec_verified,
                ..
            } = s;
            let draft = spec_draft
                .as_ref()
                .ok_or_else(|| hip_bridge::HipError::new(0, "verify: no draft"))?;
            lanes.push(MtpCbVerifyLane {
                kv_cache: kv,
                dn_state: dn,
                state: mtp.as_mut().expect("checked at provision"),
                draft,
                eos_token_id: eos,
            });
            outcomes_into.push(spec_verified);
        }
        let outcomes = mtp_cb_verify(gpu, weights, config, scratch, &engine.cb, &mut lanes)?;
        for (slot, o) in outcomes_into.into_iter().zip(outcomes) {
            *slot = Some(o);
        }
        Ok(())
    }

    /// Commit one verified request: accept/repair (shared-verify lanes) and
    /// publish its window as the singleton would — committed ids, position,
    /// pending seed, history.
    pub(super) fn spec_commit_verify(
        &mut self,
        gpu: &mut Gpu,
        weights: &Qwen35Weights,
        config: &Qwen35Config,
        scratch: &Qwen35Scratch,
        epoch: &RequestEpoch,
    ) -> Result<RequestAdvance, String> {
        if self
            .request_state(epoch)
            .is_some_and(|s| s.dflash.is_some())
        {
            return self.dflash_commit_verify(gpu, weights, config, scratch, epoch);
        }
        let eos = config.eos_token;
        let s = self
            .slots
            .iter_mut()
            .flatten()
            .find(|s| s.epoch == *epoch)
            .ok_or_else(|| format!("commit_step: epoch {epoch:?} no longer owns its slot"))?;
        let Qwen35RequestState {
            kv,
            dn,
            mtp,
            spec_draft,
            spec_verified,
            ..
        } = &mut *s;
        let draft = spec_draft
            .take()
            .ok_or("commit_step: verify without a draft")?;
        let result = match spec_verified
            .take()
            .ok_or("commit_step: verify without an outcome")?
        {
            MtpCbVerified::Done(r) => r,
            MtpCbVerified::Pending => {
                let mut lane = MtpCbVerifyLane {
                    kv_cache: kv,
                    dn_state: dn,
                    state: mtp.as_mut().ok_or("commit_step: lane lost its MTP state")?,
                    draft: &draft,
                    eos_token_id: eos,
                };
                mtp_cb_accept(gpu, weights, config, scratch, &mut lane)
                    .map_err(|e| format!("commit_step: accept: {e}"))?
            }
        };
        s.position += result.advance;
        s.pending_seed = result.committed.last().copied();
        s.history.extend_from_slice(&result.committed);
        let finish = if result.committed.iter().any(|t| s.stop_ids.contains(t)) {
            Some("stop".to_string())
        } else if s.position >= s.kv.vmm_logical_bound() {
            Some("length".to_string())
        } else {
            None
        };
        Ok(RequestAdvance {
            epoch: *epoch,
            committed_ids: result.committed,
            committed_position: s.position,
            accepted_drafts: result.accept_count,
            verified_rows: result.drafts_generated + 1,
            finish,
        })
    }

    /// Drop the drafts/outcomes of a plan's Verify requests (abort).
    pub(super) fn spec_clear_planned(&mut self, plan: &BatchStepPlan) {
        for r in &plan.requests {
            if let Some(s) = self.request_state_mut(&r.epoch) {
                s.spec_draft = None;
                s.spec_verified = None;
                s.dflash_draft = None;
                if let Some(lane) = s.dflash.as_mut() {
                    lane.verified = false;
                    lane.picks.clear();
                }
            }
        }
    }
}

/// A running singleton MTP request's drafter state at a committed window
/// boundary: head KV rows `[0, rows)` (`rows` = the request's position),
/// `prev_hidden` and its position. Device copies, independent of the
/// singleton speculator (which is reset after promotion).
pub struct MtpLaneSnapshot {
    rows: usize,
    row_bytes: usize,
    k_rows: GpuTensor,
    v_rows: GpuTensor,
    prev_hidden: GpuTensor,
    prev_hidden_pos: Option<usize>,
}

impl MtpLaneSnapshot {
    /// Copy the live drafter state of `spec` (a qwen35 MTP speculator) at
    /// committed position `rows`. `Err` when `spec` is not one or holds no
    /// Q8-head state.
    pub fn capture(gpu: &mut Gpu, spec: &mut dyn Speculator, rows: usize) -> Result<Self, String> {
        let st = spec
            .drafter_any_mut()
            .and_then(|d| d.downcast_mut::<Qwen35MtpDrafter>())
            .ok_or("speculator is not the qwen35 MTP drafter")?
            .mtp_live_state()
            .ok_or("MTP drafter has no live state")?;
        if st.mtp_kv.kv_mode != MtpKvMode::Q8 || rows > st.mtp_kv.max_seq {
            return Err(format!(
                "MTP head KV {:?} / rows {rows} > {}",
                st.mtp_kv.kv_mode, st.mtp_kv.max_seq
            ));
        }
        let row_bytes = st.mtp_kv.n_head_kv * (st.mtp_kv.head_dim / 32) * 34;
        let bytes = (rows * row_bytes).max(4);
        let mut out: Vec<GpuTensor> = Vec::with_capacity(3);
        let copied = (|| -> Result<(), String> {
            for (src, n) in [
                (&st.mtp_kv.inner.k_gpu[0], bytes),
                (&st.mtp_kv.inner.v_gpu[0], bytes),
                (&st.prev_hidden, st.prev_hidden.byte_size()),
            ] {
                let t = gpu
                    .alloc_tensor(&[n.div_ceil(4)], DType::F32)
                    .map_err(|e| e.to_string())?;
                out.push(t);
                let dst = out.last().expect("pushed");
                gpu.memcpy_dtod_at_auto(&dst.buf, 0, &src.buf, 0, n.min(src.buf.size()))
                    .map_err(|e| e.to_string())?;
            }
            gpu.hip.device_synchronize().map_err(|e| e.to_string())
        })();
        if let Err(e) = copied {
            for t in out {
                let _ = gpu.free_tensor(t);
            }
            return Err(format!("MTP snapshot: {e}"));
        }
        let prev_hidden = out.pop().expect("three tensors");
        let v_rows = out.pop().expect("three tensors");
        let k_rows = out.pop().expect("three tensors");
        Ok(Self {
            rows,
            row_bytes,
            k_rows,
            v_rows,
            prev_hidden,
            prev_hidden_pos: st.prev_hidden_pos,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.k_rows);
        let _ = gpu.free_tensor(self.v_rows);
        let _ = gpu.free_tensor(self.prev_hidden);
    }
}
