// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Recurrence-only rollback of a partially accepted DFlash2 verify window.
//!
//! The verify over a block of `rows` rows advances every RDA layer's recurrent
//! matrix and q/k/v conv history by all `rows`. A partial accept keeps only the
//! first `keep` rows. Instead of restoring the snapshot and running a second
//! full forward over those rows (projections, mHC, MLA, MoE and RDA), only what
//! the discarded rows corrupted is rebuilt:
//!
//! 1. the armed verify recorded, per RDA layer, the inputs of the recurrence
//!    (`q, k, v', log_alpha, beta`, the outputs of `raon_rda_prepare`) and the
//!    raw pre-conv projections (`q_raw, k_raw, v_raw`) of every row into the
//!    [`RdaRing`] (no copy: the forward writes those planes in place of its
//!    scratch ones, see [`crate::forward::RdaRing`]);
//! 2. [`RaonBundle::rollback_rows`] restores the persistent-slab snapshot
//!    (RDA matrices, conv histories, device `seq_len`; the mirror watermark is
//!    clamped by [`crate::state::RaonState::restore`]), reruns the layer's
//!    `execute_gated_delta` over the first `keep` recorded rows with the same
//!    geometry, strides and slot as `rda_forward_rows` (output discarded), and
//!    rebuilds the conv history as the last `K - 1 = 3` rows of
//!    `snapshot_history ++ raw_inputs[..keep]` with pitched device copies;
//! 3. the length is set to `position + keep`. The MLA latent rows past it are
//!    dead and rewritten by the next forward; mHC carries no state.
//!
//! # Conv history is the raw pre-conv input
//!
//! `causal_conv_depthwise_silu_rows_f32` (`kernels/src/causal_conv_depthwise_rows.hip:8-10`,
//! `:84-90`) forms `padded = [hist[0..K-2] (oldest -> newest), in[0..N-1]]` and
//! stores `hist_out[j, c] = padded[N + j, c]` for `j < K - 1`, with `hist_out`
//! rows `hist_tap_stride` apart (`rdna-compute/src/gated_delta_vector.rs:270-272`,
//! `:294`; `rda.rs` passes `hist_tap_stride = P`). `padded[i]` is a raw input
//! row (never a post-conv value, `:84-90` read `in[]` not `out[]`), so after
//! `keep` rows the history is `padded[keep .. keep + 3]`, the layout
//! `[conv_hist = 3, channels]` oldest -> newest of [`crate::state::RaonState::conv_q`].
//!
//! # Switch
//!
//! The process default is `recurrence`; the rollback-mode environment variable
//! (see `docs/env-vars.md`, Raon section) set to `replay` selects the previous
//! behavior (restore + full eager replay through the batched forward). It is
//! read once per process through the `[developer]` namespace;
//! [`RaonBundle::set_dflash_rollback_mode`] overrides it per bundle (oracle).

use std::sync::LazyLock;

use rdna_compute::{DType, Gpu, GpuTensor};

use crate::bundle::RaonBundle;
use crate::forward::{forward_rows_verify, RdaRing};
use crate::rda::{rda_recurrence, RdaGeometry, RdaLayerState};
use crate::spec_impl::RaonSpecState;
use crate::state::RaonStateSnapshot;

/// Environment variable selecting the partial-accept rollback ([`RollbackMode`]).
pub const ROLLBACK_ENV: &str = "HIPFIRE_RAON_DFLASH_ROLLBACK";

/// How a partial accept repairs the over-advanced recurrent state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RollbackMode {
    /// Restore the snapshot, rerun only the RDA recurrence over the kept rows
    /// from the verify's recorded inputs, rebuild the conv history (default).
    Recurrence,
    /// Restore the snapshot and replay the kept rows through the full eager
    /// batched forward (the kill switch).
    Replay,
}

impl RollbackMode {
    /// Parse the environment value: unset/empty/`recurrence` or `replay`;
    /// anything else is a named error, never a silent fallback.
    pub fn parse(value: Option<&str>) -> Result<Self, String> {
        match value.map(|v| v.trim().to_ascii_lowercase()).as_deref() {
            None | Some("") | Some("recurrence") => Ok(Self::Recurrence),
            Some("replay") => Ok(Self::Replay),
            Some(other) => Err(format!(
                "{ROLLBACK_ENV}={other:?}: expected `recurrence` or `replay`"
            )),
        }
    }

    /// The process default, read once.
    pub fn process_default() -> Result<Self, String> {
        static MODE: LazyLock<Result<RollbackMode, String>> = LazyLock::new(|| {
            RollbackMode::parse(hipfire_config::developer_var(ROLLBACK_ENV).ok().as_deref())
        });
        MODE.clone()
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Recurrence => "recurrence",
            Self::Replay => "replay",
        }
    }
}

/// Owner of the per-RDA-layer recurrence-input ring: `layers[ri]` holds
/// `max_rows` rows of every plane ([`RdaRing::layer_elems`]). `ticket` is the
/// `(position, rows)` of the verify whose inputs it holds.
pub(crate) struct RdaRingBuf {
    pub(crate) max_rows: usize,
    pub(crate) layers: Vec<GpuTensor>,
    pub(crate) ticket: Option<(usize, usize)>,
}

impl RdaRingBuf {
    fn alloc(
        gpu: &mut Gpu,
        geom: &RdaGeometry,
        n_rda: usize,
        max_rows: usize,
    ) -> Result<Self, String> {
        let elems = RdaRing::layer_elems(max_rows, geom.proj(), geom.n_heads);
        let mut layers: Vec<GpuTensor> = Vec::with_capacity(n_rda);
        for _ in 0..n_rda {
            match gpu.alloc_tensor(&[elems], DType::F32) {
                Ok(t) => layers.push(t),
                Err(e) => {
                    for t in layers {
                        let _ = gpu.free_tensor(t);
                    }
                    return Err(format!(
                        "raon1_mini: alloc RDA rollback ring ({n_rda} x {elems} F32): {e:?}"
                    ));
                }
            }
        }
        Ok(Self {
            max_rows,
            layers,
            ticket: None,
        })
    }
}

impl RaonSpecState {
    /// The forward-side ring handle (`None` until [`RaonBundle::ensure_rda_ring`]).
    pub(crate) fn ring(&self) -> Option<RdaRing<'_>> {
        self.ring.as_ref().map(|r| RdaRing {
            layers: &r.layers,
            max_rows: r.max_rows,
        })
    }

    /// Graph-key word of the ring: every layer's address and the row capacity
    /// (0 = no ring). Changes only when [`RaonBundle::ensure_rda_ring`] reallocates.
    pub(crate) fn ring_binding(&self) -> u64 {
        let Some(ring) = self.ring() else {
            return 0;
        };
        let mut words = Vec::new();
        ring.binding_words(&mut words);
        words.into_iter().fold(0xcbf2_9ce4_8422_2325u64, |h, w| {
            (h ^ w).wrapping_mul(0x0100_0000_01b3)
        }) | 1
    }

    /// The ring now holds the inputs of the verify of `rows` rows at `position`
    /// (an armed eager verify records it; a replayed graph calls this itself).
    pub(crate) fn mark_ring_recorded(&mut self, position: usize, rows: usize) {
        if let Some(r) = self.ring.as_mut() {
            r.ticket = Some((position, rows));
        }
    }

    pub(crate) fn clear_ring_ticket(&mut self) {
        if let Some(r) = self.ring.as_mut() {
            r.ticket = None;
        }
    }

    pub(crate) fn ring_ticket(&self) -> Option<(usize, usize)> {
        self.ring.as_ref().and_then(|r| r.ticket)
    }
}

impl RaonBundle {
    /// Override the partial-accept rollback of this bundle (the process default
    /// comes from [`ROLLBACK_ENV`]). Takes effect for the next verify: the ring
    /// is armed by the verify, so a mode switched between a verify and its
    /// commit is refused by [`Self::rollback_rows`].
    pub fn set_dflash_rollback_mode(&mut self, mode: RollbackMode) {
        self.spec.rollback = Some(mode);
    }

    /// The effective rollback mode: the per-bundle override, else the process default.
    pub fn dflash_rollback_mode(&self) -> Result<RollbackMode, String> {
        match self.spec.rollback {
            Some(m) => Ok(m),
            None => RollbackMode::process_default(),
        }
    }

    /// Size the recurrence-input ring for verify blocks of up to `rows` rows
    /// (`new_spec_scratch`). Reallocates (moving every address) only when it
    /// must grow.
    pub(crate) fn ensure_rda_ring(&mut self, gpu: &mut Gpu, rows: usize) -> Result<(), String> {
        if self.spec.ring.as_ref().is_some_and(|r| r.max_rows >= rows) {
            return Ok(());
        }
        let geom = *self.work.rda_geom();
        let ring = RdaRingBuf::alloc(gpu, &geom, self.config.n_rda(), rows)?;
        if let Some(old) = self.spec.ring.replace(ring) {
            for t in old.layers {
                let _ = gpu.free_tensor(t);
            }
        }
        Ok(())
    }

    /// One verify forward over `block` at the slot's current position: the
    /// speculative `forward_chunk(.., want_logits = false, capture_row0)` with
    /// the rollback ring armed when the mode is [`RollbackMode::Recurrence`]
    /// (the launch sequence is the unarmed one; only the RDA pointers move).
    /// Records the ring ticket `(position, block.len())` on success.
    pub(crate) fn forward_verify_chunk(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        position: usize,
        capture_row0: Option<usize>,
    ) -> Result<(), String> {
        let arm = self.dflash_rollback_mode()? == RollbackMode::Recurrence;
        if arm && self.spec.ring.is_none() {
            return Err("raon1_mini verify: the RDA rollback ring is not allocated \
                        (new_spec_scratch not called)"
                .into());
        }
        self.spec.clear_ring_ticket();
        let pos = self.state.n_tokens() as u32;
        {
            let capture = capture_row0.map(|r| self.spec.capture(r)).transpose()?;
            let ring = if arm { self.spec.ring() } else { None };
            forward_rows_verify(
                &self.config,
                &self.weights,
                &mut self.state,
                &mut self.scratch,
                &mut self.work,
                gpu,
                block,
                pos,
                false,
                capture.as_ref(),
                ring.as_ref(),
            )?;
        }
        if arm {
            self.spec.mark_ring_recorded(position, block.len());
        }
        Ok(())
    }

    /// Roll a verify of `rows` rows at `position` back to its first `keep`
    /// rows (`1..rows`): restore `snapshot`, rerun every RDA layer's recurrence
    /// over the recorded inputs of the kept rows, rebuild the conv histories,
    /// set the length to `position + keep`. See the module documentation.
    pub(crate) fn rollback_rows(
        &mut self,
        gpu: &mut Gpu,
        snapshot: &RaonStateSnapshot,
        position: usize,
        rows: usize,
        keep: usize,
    ) -> Result<(), String> {
        if keep == 0 || keep >= rows {
            return Err(format!(
                "raon1_mini rollback: keep {keep} outside 1..{rows} rows"
            ));
        }
        if self.spec.ring_ticket() != Some((position, rows)) {
            return Err(format!(
                "raon1_mini rollback: the RDA ring holds {:?}, not the verify of {rows} rows at \
                 {position} (was it verified under {} = recurrence?)",
                self.spec.ring_ticket(),
                ROLLBACK_ENV
            ));
        }
        let geom = *self.work.rda_geom();
        let proj = geom.proj();
        let hist = geom.conv_kernel - 1;
        let layout = self.state.conv_slab_layout()?;
        if layout.cols != proj || layout.hist != hist {
            return Err(format!(
                "raon1_mini rollback: conv history is [{}, {}] per plane, RDA needs [{hist}, {proj}]",
                layout.hist, layout.cols
            ));
        }

        self.state.restore(gpu, snapshot)?;

        let ring = self.spec.ring().expect("ring ticket implies a ring");
        let row_bytes = proj * 4;
        let ring_plane_bytes = ring.max_rows * row_bytes;
        // Rows of the kept inputs that enter the new history.
        let m = keep.min(hist);
        let discard = self.work.rda_discard();
        for ri in 0..self.config.n_rda() {
            let tape = ring.tape(ri, &geom);
            let st = RdaLayerState {
                state: &self.state.rda[ri],
                hist_q: &self.state.conv_q[ri],
                hist_k: &self.state.conv_k[ri],
                hist_v: &self.state.conv_v[ri],
                slot: 0,
                state_slot_stride: geom.state_elems(),
            };
            rda_recurrence(
                gpu,
                &geom,
                &st,
                &tape.q,
                &tape.k,
                &tape.vprime,
                &tape.log_alpha,
                &tape.beta,
                discard,
                keep,
            )
            .map_err(|e| format!("raon1_mini rollback L-rda{ri} recurrence: {e}"))?;

            // New history rows [0, hist) = padded[keep .. keep + hist] with
            // padded = snapshot_history ++ raw_inputs: the old rows [keep, hist)
            // (only when keep < hist) then the last `m` kept raw rows.
            let base = layout.conv_off + ri * layout.layer_stride;
            let copy_err = |what: &str, e: rdna_compute::HipError| {
                format!("raon1_mini rollback RDA layer {ri} conv {what}: {e:?}")
            };
            if keep < hist {
                gpu.memcpy_dtod_2d_at_ordered_async(
                    &self.state.persist_tensor().buf,
                    base,
                    layout.plane_pitch,
                    &snapshot.tensor().buf,
                    base + keep * row_bytes,
                    layout.plane_pitch,
                    (hist - keep) * row_bytes,
                    3,
                )
                .map_err(|e| copy_err("old rows", e))?;
            }
            gpu.memcpy_dtod_2d_at_ordered_async(
                &self.state.persist_tensor().buf,
                base + (hist - m) * row_bytes,
                layout.plane_pitch,
                &ring.layers[ri].buf,
                (keep - m) * row_bytes,
                ring_plane_bytes,
                m * row_bytes,
                3,
            )
            .map_err(|e| copy_err("raw rows", e))?;
        }
        self.state.set_n_tokens(gpu, position + keep)
    }
}
