// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Device accept / residual / bonus of a sampled singleton DFlash window
//! (`kernels/src/dflash_sampled_accept_sparse_q.hip`).
//!
//! The target rows are the resident output of
//! [`Gpu::softmax_temp_topp_batched_into_f32`] (`probs`, per-row `tau` and
//! `z`); the host uploads ONE payload (proposed tokens, pre-drawn uniforms,
//! selector q as sparse rows) and downloads four words. Nothing vocab-sized
//! crosses PCIe and no dense q exists. See the kernel header for the law, the
//! draw order and the fallbacks.
//!
//! Self-contained via `include_str!` (no shared-registry edits), launched
//! through `launch_maybe_blob` like the other sampled-row kernels.
//!
//! [`dflash_sampled_accept_reference`] is the bit-exact host twin of the
//! kernel (same segment association), for oracles and the host-only tests.

use crate::dispatch::{DType, Gpu, GpuTensor};
use hip_bridge::{HipError, HipResult, KernargBlob};
use std::ffi::c_void;

/// Kernel source for [`Gpu::dflash_sampled_accept_sparse_q_f32`].
pub const DFLASH_SAMPLED_ACCEPT_SRC: &str =
    include_str!("../../../kernels/src/dflash_sampled_accept_sparse_q.hip");
/// Module key.
pub const DFLASH_SAMPLED_ACCEPT_MODULE: &str = "dflash_sampled_accept_sparse_q";
/// Device symbol.
pub const DFLASH_SAMPLED_ACCEPT_SYMBOL: &str = "dflash_sampled_accept_sparse_q_f32";
/// Threads (= CDF segments) of the single block; `DSA_THREADS` in the kernel.
pub const DFLASH_SAMPLED_ACCEPT_THREADS: usize = 256;
/// Certified builder (PeaceMaker) module of the production gfx1201 kernel
/// (`peacemaker-isa emit --kernel dflash_accept --arch gfx1201`), same ABI, grid
/// and output words as the hipcc oracle.
pub const DFLASH_SAMPLED_ACCEPT_PM_MODULE: &str = "dflash_sampled_accept_pm_gfx1201";
/// Device symbol of the production builder kernel.
pub const DFLASH_SAMPLED_ACCEPT_PM_SYMBOL: &str = "dflash_sampled_accept_sparse_q_f32_pm_gfx1201";
/// Largest probs extent the builder kernel addresses (32-bit raw buffer offsets).
const DFLASH_ACCEPT_PM_MAX_PROBS_BYTES: u64 = 1 << 32;

/// The production device accept runs from the certified builder module on
/// gfx1201 only; there is no hipcc production fallback (the hipcc kernel is the
/// retained oracle of [`Gpu::dflash_sampled_accept_sparse_q_f32_hip_oracle`]).
pub fn dflash_accept_pm_arch(arch: &str) -> bool {
    arch == "gfx1201"
}

/// The builder kernel's shape contract: at least one proposed row and one
/// vocabulary id, and the `(n + 1)`-row probs buffer addressable by 32-bit
/// byte offsets.
pub fn dflash_accept_pm_fits(n: usize, vocab: usize) -> bool {
    n >= 1
        && vocab >= 1
        && vocab <= i32::MAX as usize
        && n <= i32::MAX as usize / 4
        && ((n as u64 + 1) * vocab as u64 * 4) < DFLASH_ACCEPT_PM_MAX_PROBS_BYTES
}

/// Which kernel a launch uses.
#[derive(Clone, Copy, PartialEq, Eq)]
enum AcceptKernel {
    /// Production: the certified builder module.
    Pm,
    /// Non-production oracle: the hipcc source.
    HipOracle,
}

const FLT_MIN: f32 = f32::MIN_POSITIVE;

fn err(msg: impl AsRef<str>) -> HipError {
    HipError::new(
        0,
        &format!("dflash_sampled_accept_sparse_q_f32: {}", msg.as_ref()),
    )
}

/// Decision of one window: the first `accept_len` proposed tokens are
/// accepted, `bonus` follows them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DflashDeviceDecision {
    pub accept_len: usize,
    pub bonus: u32,
    /// A proposed token was rejected inside the budget (`bonus` is the residual draw).
    pub rejected: bool,
    /// 0 normal draw; 1 degenerate residual (argmax of the target row); 2 the
    /// draw fell off the end of the CDF (last support id).
    pub fallback: u32,
}

/// Number of payload words of `n` draft rows with `nnz` q entries.
pub fn dflash_accept_payload_words(n: usize, nnz: usize) -> usize {
    4 * n + 3 + 2 * nnz
}

/// Pack a window into the kernel payload (layout in the kernel header).
///
/// * `tokens` (`n`): the proposed tokens `block[1..]`.
/// * `uniforms` (`n`): one accept uniform per row, `u_fin`: the single
///   residual-or-bonus uniform; every one drawn from `[0, 1)` by the host RNG.
/// * `max_accept`: rows allowed to be accepted (clamped to `n`).
/// * `selected_q` (`n`): q at the proposed token.
/// * `q_off` (`n + 1`), `q_idx`, `q_val`: the sparse selector q rows (ascending
///   ids per row, last-write already applied).
#[allow(clippy::too_many_arguments)]
pub fn dflash_accept_pack_payload(
    out: &mut Vec<u32>,
    tokens: &[u32],
    uniforms: &[f32],
    u_fin: f32,
    max_accept: usize,
    selected_q: &[f32],
    q_off: &[u32],
    q_idx: &[u32],
    q_val: &[f32],
) -> Result<(), String> {
    let n = tokens.len();
    if uniforms.len() != n || selected_q.len() != n || q_off.len() != n + 1 {
        return Err(format!(
            "payload shape: {n} tokens, {} uniforms, {} selected q, {} offsets",
            uniforms.len(),
            selected_q.len(),
            q_off.len()
        ));
    }
    let nnz = q_off[n] as usize;
    if q_off[0] != 0
        || q_off.windows(2).any(|w| w[0] > w[1])
        || q_idx.len() != nnz
        || q_val.len() != nnz
    {
        return Err(format!(
            "payload q rows: offsets {:?}.., {} ids, {} values for {nnz} entries",
            &q_off[..q_off.len().min(3)],
            q_idx.len(),
            q_val.len()
        ));
    }
    out.clear();
    out.reserve(dflash_accept_payload_words(n, nnz));
    out.extend_from_slice(tokens);
    out.extend(uniforms.iter().map(|u| u.to_bits()));
    out.push(u_fin.to_bits());
    out.push(max_accept.min(n) as u32);
    out.extend(selected_q.iter().map(|q| q.to_bits()));
    out.extend_from_slice(q_off);
    out.extend_from_slice(q_idx);
    out.extend(q_val.iter().map(|v| v.to_bits()));
    debug_assert_eq!(out.len(), dflash_accept_payload_words(n, nnz));
    Ok(())
}

/// Check `payload` is a well-formed window of `n` rows over `vocab` ids.
fn validate_payload(payload: &[u32], n: usize, vocab: usize) -> Result<(), String> {
    if payload.len() < 4 * n + 3 {
        return Err(format!(
            "payload has {} words, header alone needs {}",
            payload.len(),
            4 * n + 3
        ));
    }
    let off = &payload[3 * n + 2..4 * n + 3];
    let nnz = off[n] as usize;
    if payload.len() != dflash_accept_payload_words(n, nnz) {
        return Err(format!(
            "payload has {} words, {n} rows with {nnz} q entries need {}",
            payload.len(),
            dflash_accept_payload_words(n, nnz)
        ));
    }
    if off[0] != 0 || off.windows(2).any(|w| w[0] > w[1]) {
        return Err("q offsets are not a non-decreasing run from 0".into());
    }
    if payload[..n].iter().any(|&t| t as usize >= vocab) {
        return Err(format!("a proposed token is outside vocab {vocab}"));
    }
    let ids = &payload[4 * n + 3..4 * n + 3 + nnz];
    for r in 0..n {
        let row = &ids[off[r] as usize..off[r + 1] as usize];
        if row.iter().any(|&i| i as usize >= vocab) || row.windows(2).any(|w| w[0] >= w[1]) {
            return Err(format!(
                "q row {r} ids are not strictly ascending and < vocab"
            ));
        }
    }
    Ok(())
}

#[inline]
fn eff(p: f32, tau: f32, inv_z: f32) -> f32 {
    if tau <= 0.0 {
        p
    } else if p >= tau {
        p * inv_z
    } else {
        0.0
    }
}

/// Bit-exact host twin of the kernel: same accept tests, same 256-segment CDF
/// association, same fallbacks. `probs` is `[(n + 1) * vocab]` raw softmax
/// rows, `tau` / `z` are `[n + 1]`.
pub fn dflash_sampled_accept_reference(
    probs: &[f32],
    tau: &[f32],
    z: &[f32],
    payload: &[u32],
    n: usize,
    vocab: usize,
) -> DflashDeviceDecision {
    let tokens = &payload[..n];
    let uacc = &payload[n..2 * n];
    let u_fin = f32::from_bits(payload[2 * n]);
    let max_accept = payload[2 * n + 1] as usize;
    let selq = &payload[2 * n + 2..3 * n + 2];
    let qoff = &payload[3 * n + 2..4 * n + 3];
    let nnz = qoff[n] as usize;
    let qids = &payload[4 * n + 3..4 * n + 3 + nnz];
    let qvals = &payload[4 * n + 3 + nnz..];

    let inv_z_of = |r: usize| 1.0f32 / z[r].max(FLT_MIN);

    let lim = n.min(max_accept);
    let mut first_rej = None;
    for i in 0..lim {
        let p = probs[i * vocab + tokens[i] as usize];
        let p_t = eff(p, tau[i], inv_z_of(i));
        let p_d = f32::from_bits(selq[i]).max(FLT_MIN);
        let u = f32::from_bits(uacc[i]);
        if !(u * p_d <= p_t) {
            first_rej = Some(i);
            break;
        }
    }
    let rejected = first_rej.is_some();
    let row = first_rej.unwrap_or(lim);

    let t_row = tau[row];
    let inv_z = inv_z_of(row);
    let prow = &probs[row * vocab..(row + 1) * vocab];
    let (q_ids, q_vals): (&[u32], &[u32]) = if rejected {
        let (a, b) = (qoff[row] as usize, qoff[row + 1] as usize);
        (&qids[a..b], &qvals[a..b])
    } else {
        (&[], &[])
    };

    const T: usize = DFLASH_SAMPLED_ACCEPT_THREADS;
    let seg = vocab.div_ceil(T);
    let bounds = |t: usize| {
        let s = (t * seg).min(vocab);
        (s, (s + seg).min(vocab))
    };
    // Summand of column `c` given the first q entry not yet consumed.
    let summand = |c: usize, qp: &mut usize| -> f32 {
        let mut d = eff(prow[c], t_row, inv_z);
        if rejected {
            let mut q = 0.0f32;
            if *qp < q_ids.len() && q_ids[*qp] as usize == c {
                q = f32::from_bits(q_vals[*qp]);
                *qp += 1;
            }
            d -= q;
        }
        d
    };

    let mut seg_sum = [0.0f32; T];
    let mut seg_last = [-1i64; T];
    for t in 0..T {
        let (s, e) = bounds(t);
        let mut qp = q_ids.partition_point(|&i| (i as usize) < s);
        let mut l = 0.0f32;
        for c in s..e {
            let d = summand(c, &mut qp);
            if d > 0.0 {
                l += d;
                seg_last[t] = c as i64;
            }
        }
        seg_sum[t] = l;
    }
    let mut base = [0.0f32; T + 1];
    let mut acc = 0.0f32;
    for t in 0..T {
        acc += seg_sum[t];
        base[t + 1] = acc;
    }
    let total = acc;

    let mut fallback = 0u32;
    let us;
    if rejected && total <= 0.0 {
        // Degenerate: first maximum of the truncated target row.
        let mut best_v = f32::NEG_INFINITY;
        let mut best_i = 0usize;
        for c in 0..vocab {
            let v = eff(prow[c], t_row, inv_z);
            if v > best_v {
                best_v = v;
                best_i = c;
            }
        }
        return DflashDeviceDecision {
            accept_len: row,
            bonus: best_i as u32,
            rejected,
            fallback: 1,
        };
    } else if rejected {
        us = u_fin * total;
    } else {
        us = u_fin;
    }

    let mut bonus: Option<usize> = None;
    for t in 0..T {
        if base[t] <= us && us < base[t + 1] {
            let (s, e) = bounds(t);
            let mut qp = q_ids.partition_point(|&i| (i as usize) < s);
            let mut l = 0.0f32;
            for c in s..e {
                let d = summand(c, &mut qp);
                if d > 0.0 {
                    l += d;
                    if us < base[t] + l {
                        bonus = Some(c);
                        break;
                    }
                }
            }
            break;
        }
    }
    let bonus = bonus.unwrap_or_else(|| {
        fallback = 2;
        (0..T)
            .rev()
            .find_map(|t| (seg_last[t] >= 0).then(|| seg_last[t] as usize))
            .unwrap_or(vocab - 1)
    });
    DflashDeviceDecision {
        accept_len: row,
        bonus: bonus as u32,
        rejected,
        fallback,
    }
}

/// Reusable device buffers of the window accept: the verify rows' softmax
/// (`probs`, `tau`, `z`: the caller enqueues
/// [`Gpu::softmax_temp_topp_batched_into_f32`] into them), the payload and the
/// four result words. Grow-only ([`Self::ensure`]); a session owns one, so a
/// steady-state window allocates nothing.
pub struct DflashAcceptScratch {
    pub probs: GpuTensor,
    pub tau: GpuTensor,
    pub z: GpuTensor,
    payload: GpuTensor,
    out: GpuTensor,
    rows_cap: usize,
    vocab_cap: usize,
    payload_cap: usize,
    /// Host staging reused every window (pre-drawn uniforms, packed payload).
    pub host_uniforms: Vec<f32>,
    pub host_payload: Vec<u32>,
}

impl DflashAcceptScratch {
    /// Scratch for windows of up to `rows` verify rows over `vocab` ids and
    /// `payload_words` payload words.
    pub fn new(gpu: &mut Gpu, rows: usize, vocab: usize, payload_words: usize) -> HipResult<Self> {
        let probs = gpu.alloc_tensor(&[rows * vocab], DType::F32)?;
        let tau = match gpu.alloc_tensor(&[rows], DType::F32) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.free_tensor(probs);
                return Err(e);
            }
        };
        let z = match gpu.alloc_tensor(&[rows], DType::F32) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.free_tensor(probs);
                let _ = gpu.free_tensor(tau);
                return Err(e);
            }
        };
        let payload = match gpu.alloc_tensor(&[payload_words.max(1)], DType::F32) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.free_tensor(probs);
                let _ = gpu.free_tensor(tau);
                let _ = gpu.free_tensor(z);
                return Err(e);
            }
        };
        let out = match gpu.alloc_tensor(&[4], DType::F32) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.free_tensor(probs);
                let _ = gpu.free_tensor(tau);
                let _ = gpu.free_tensor(z);
                let _ = gpu.free_tensor(payload);
                return Err(e);
            }
        };
        Ok(Self {
            probs,
            tau,
            z,
            payload,
            out,
            rows_cap: rows,
            vocab_cap: vocab,
            payload_cap: payload_words.max(1),
            host_uniforms: Vec::with_capacity(rows),
            host_payload: Vec::with_capacity(payload_words),
        })
    }

    /// Grow to hold a `rows` x `vocab` window with `payload_words` payload
    /// words; a no-op when it already does. The previous buffers are
    /// released only after the new ones exist, and only between windows (the
    /// result readback synchronizes, so nothing still uses them).
    pub fn ensure(
        slot: &mut Option<Self>,
        gpu: &mut Gpu,
        rows: usize,
        vocab: usize,
        payload_words: usize,
    ) -> HipResult<()> {
        if let Some(s) = slot.as_ref() {
            if s.rows_cap >= rows && s.vocab_cap >= vocab && s.payload_cap >= payload_words {
                return Ok(());
            }
        }
        let (r, v, p) = match slot.as_ref() {
            Some(s) => (
                s.rows_cap.max(rows),
                s.vocab_cap.max(vocab),
                s.payload_cap.max(payload_words),
            ),
            None => (rows, vocab, payload_words),
        };
        let new = Self::new(gpu, r, v, p)?;
        if let Some(old) = slot.replace(new) {
            old.free(gpu);
        }
        Ok(())
    }

    /// Release the buffers.
    pub fn free(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.probs);
        let _ = gpu.free_tensor(self.tau);
        let _ = gpu.free_tensor(self.z);
        let _ = gpu.free_tensor(self.payload);
        let _ = gpu.free_tensor(self.out);
    }
}

impl Gpu {
    /// Decide a sampled DFlash window on the device.
    ///
    /// `probs` / `tau` / `z` are the resident output of
    /// [`Self::softmax_temp_topp_batched_into_f32`] over `n + 1` verify rows
    /// (`probs` `[(n + 1) * vocab]`, `tau`/`z` `[n + 1]`); `payload` is
    /// [`dflash_accept_pack_payload`]'s window of `n` draft rows. The payload
    /// upload and the launch are enqueued on the Gpu's stream, behind the
    /// softmax; the single synchronizing step is the 16-byte result readback.
    /// No capture-safe: the readback refuses to run during capture.
    ///
    /// Production route: the certified builder kernel
    /// ([`DFLASH_SAMPLED_ACCEPT_PM_SYMBOL`]), gfx1201 only. There is no hipcc
    /// fallback: other architectures (and shapes outside
    /// [`dflash_accept_pm_fits`]) are an error, and callers gate on
    /// [`Gpu::dflash_device_accept_supported`].
    #[allow(clippy::too_many_arguments)]
    pub fn dflash_sampled_accept_sparse_q_f32(
        &mut self,
        scratch: &DflashAcceptScratch,
        probs: &GpuTensor,
        tau: &GpuTensor,
        z: &GpuTensor,
        payload: &[u32],
        n: usize,
        vocab: usize,
    ) -> HipResult<DflashDeviceDecision> {
        self.dflash_sampled_accept_run(AcceptKernel::Pm, scratch, probs, tau, z, payload, n, vocab)
    }

    /// Whether the production device accept can run on this device (the
    /// builder module exists for gfx1201 only).
    pub fn dflash_device_accept_supported(&self) -> bool {
        dflash_accept_pm_arch(&self.arch)
    }

    /// The retained hipcc kernel (`kernels/src/dflash_sampled_accept_sparse_q.hip`)
    /// as a NON-PRODUCTION oracle for the builder kernel's equality tests; the
    /// decision words are identical. No production path calls it.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn dflash_sampled_accept_sparse_q_f32_hip_oracle(
        &mut self,
        scratch: &DflashAcceptScratch,
        probs: &GpuTensor,
        tau: &GpuTensor,
        z: &GpuTensor,
        payload: &[u32],
        n: usize,
        vocab: usize,
    ) -> HipResult<DflashDeviceDecision> {
        self.dflash_sampled_accept_run(
            AcceptKernel::HipOracle,
            scratch,
            probs,
            tau,
            z,
            payload,
            n,
            vocab,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn dflash_sampled_accept_run(
        &mut self,
        kernel: AcceptKernel,
        scratch: &DflashAcceptScratch,
        probs: &GpuTensor,
        tau: &GpuTensor,
        z: &GpuTensor,
        payload: &[u32],
        n: usize,
        vocab: usize,
    ) -> HipResult<DflashDeviceDecision> {
        if n == 0 || vocab == 0 {
            return Err(err(format!("n {n} and vocab {vocab} must be > 0")));
        }
        if vocab > i32::MAX as usize || n > i32::MAX as usize / 4 {
            return Err(err("n / vocab exceed the addressable range"));
        }
        validate_payload(payload, n, vocab).map_err(err)?;
        if probs.dtype != DType::F32 || tau.dtype != DType::F32 || z.dtype != DType::F32 {
            return Err(err(format!(
                "probs, tau and z must be F32 (got {:?}, {:?}, {:?})",
                probs.dtype, tau.dtype, z.dtype
            )));
        }
        let rows = n + 1;
        for (name, t, need) in [
            ("probs", probs, rows * vocab * 4),
            ("tau", tau, rows * 4),
            ("z", z, rows * 4),
        ] {
            if t.buf.size() < need {
                return Err(err(format!(
                    "{name} buffer too small (have {} need {need} bytes)",
                    t.buf.size()
                )));
            }
        }
        self.bind_thread()?;
        let symbol = match kernel {
            AcceptKernel::Pm => {
                if !dflash_accept_pm_arch(&self.arch) {
                    return Err(err(format!(
                        "the production device accept is a gfx1201 builder module (device is {}); no hipcc fallback",
                        self.arch
                    )));
                }
                if !dflash_accept_pm_fits(n, vocab) {
                    return Err(err(format!(
                        "n {n} x vocab {vocab} outside the builder kernel's 32-bit probs extent"
                    )));
                }
                self.ensure_embedded_kernel(
                    DFLASH_SAMPLED_ACCEPT_PM_MODULE,
                    crate::kernels::DFLASH_SAMPLED_ACCEPT_PM_GFX1201,
                    DFLASH_SAMPLED_ACCEPT_PM_SYMBOL,
                )?;
                DFLASH_SAMPLED_ACCEPT_PM_SYMBOL
            }
            AcceptKernel::HipOracle => {
                self.ensure_kernel(
                    DFLASH_SAMPLED_ACCEPT_MODULE,
                    DFLASH_SAMPLED_ACCEPT_SRC,
                    DFLASH_SAMPLED_ACCEPT_SYMBOL,
                )?;
                DFLASH_SAMPLED_ACCEPT_SYMBOL
            }
        };
        if scratch.payload_cap < payload.len() {
            return Err(err(format!(
                "scratch payload holds {} words, window needs {}",
                scratch.payload_cap,
                payload.len()
            )));
        }
        let result = self.dflash_sampled_accept_enqueue(
            symbol,
            probs,
            tau,
            z,
            payload,
            &scratch.payload,
            &scratch.out,
            n,
            vocab,
        );
        if result.is_err() {
            // An async copy of `payload` may still be queued: drain it before
            // the caller's host buffer can go away.
            let _ = self.hip.device_synchronize();
        }
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn dflash_sampled_accept_enqueue(
        &mut self,
        symbol: &'static str,
        probs: &GpuTensor,
        tau: &GpuTensor,
        z: &GpuTensor,
        payload: &[u32],
        payload_dev: &GpuTensor,
        out_dev: &GpuTensor,
        n: usize,
        vocab: usize,
    ) -> HipResult<DflashDeviceDecision> {
        // SAFETY: `payload` is a live `&[u32]`; any bytes view of it is valid.
        let bytes =
            unsafe { std::slice::from_raw_parts(payload.as_ptr() as *const u8, payload.len() * 4) };
        match self.active_stream.as_ref() {
            Some(stream) => self
                .hip
                .memcpy_htod_async(&payload_dev.buf, bytes, stream)?,
            None => self
                .hip
                .memcpy_htod_async_default(&payload_dev.buf, bytes)?,
        }
        let pp = probs.buf.as_ptr();
        let tp = tau.buf.as_ptr();
        let zp = z.buf.as_ptr();
        let yp = payload_dev.buf.as_ptr();
        let op = out_dev.buf.as_ptr();
        let n_i = n as i32;
        let vocab_i = vocab as i32;
        let mut params: [*mut c_void; 7] = [
            &pp as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &zp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &n_i as *const _ as *mut c_void,
            &vocab_i as *const _ as *mut c_void,
        ];
        let timer = crate::profile::begin_timer(
            &self.hip,
            "dflash_sampled_accept",
            symbol,
            payload.len() * 4 + 16,
        );
        let launched = self.launch_maybe_blob(
            symbol,
            [1, 1, 1],
            [DFLASH_SAMPLED_ACCEPT_THREADS as u32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(pp);
                b.push_ptr(tp);
                b.push_ptr(zp);
                b.push_ptr(yp);
                b.push_ptr(op);
                b.push_i32(n_i);
                b.push_i32(vocab_i);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        launched?;
        let mut raw = [0i32; 4];
        // SAFETY: `raw` is 16 initialized bytes; any pattern is a valid i32.
        let dst = unsafe { std::slice::from_raw_parts_mut(raw.as_mut_ptr() as *mut u8, 16) };
        self.memcpy_dtoh_auto(dst, &out_dev.buf)?;
        if raw[0] < 0 || raw[0] as usize > n || raw[1] < 0 || raw[1] as usize >= vocab {
            return Err(err(format!(
                "device result out of range: {raw:?} (n {n}, vocab {vocab})"
            )));
        }
        Ok(DflashDeviceDecision {
            accept_len: raw[0] as usize,
            bonus: raw[1] as u32,
            rejected: raw[2] != 0,
            fallback: raw[3] as u32,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(s: &mut u64) -> f32 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*s >> 40) as f32) / (1u64 << 24) as f32
    }

    /// `rows` softmax-like rows with a top-`keep` nucleus: raw probs, tau, z.
    fn rows(
        vocab: usize,
        n_rows: usize,
        keep: usize,
        seed: &mut u64,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut probs = Vec::new();
        let (mut tau, mut z) = (Vec::new(), Vec::new());
        for _ in 0..n_rows {
            let mut r: Vec<f32> = (0..vocab).map(|_| lcg(seed).powi(8) + 1e-7).collect();
            let s: f32 = r.iter().sum();
            r.iter_mut().for_each(|v| *v /= s);
            let mut sorted = r.clone();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let t = if keep == 0 { 0.0 } else { sorted[keep - 1] };
            let zz: f32 = if keep == 0 {
                1.0
            } else {
                r.iter().filter(|&&p| p >= t).sum()
            };
            probs.extend(r);
            tau.push(t);
            z.push(zz);
        }
        (probs, tau, z)
    }

    fn pack(
        tokens: &[u32],
        us: &[f32],
        u_fin: f32,
        max_accept: usize,
        selq: &[f32],
        q: &[Vec<(u32, f32)>],
    ) -> Vec<u32> {
        let mut off = vec![0u32];
        let (mut ids, mut vals) = (Vec::new(), Vec::new());
        for row in q {
            for &(i, v) in row {
                ids.push(i);
                vals.push(v);
            }
            off.push(ids.len() as u32);
        }
        let mut out = Vec::new();
        dflash_accept_pack_payload(
            &mut out, tokens, us, u_fin, max_accept, selq, &off, &ids, &vals,
        )
        .unwrap();
        out
    }

    /// The legacy host decision (ordered dense loop) over the same uniforms.
    fn legacy(
        probs: &[f32],
        tau: &[f32],
        z: &[f32],
        tokens: &[u32],
        us: &[f32],
        u_fin: f32,
        max_accept: usize,
        selq: &[f32],
        q: &[Vec<(u32, f32)>],
        vocab: usize,
    ) -> (usize, u32) {
        let n = tokens.len();
        let trunc = |r: usize| -> Vec<f32> {
            let inv = 1.0f32 / z[r].max(FLT_MIN);
            probs[r * vocab..(r + 1) * vocab]
                .iter()
                .map(|&p| eff(p, tau[r], inv))
                .collect()
        };
        for i in 0..n.min(max_accept) {
            let p = trunc(i);
            let p_d = selq[i].max(FLT_MIN);
            if !(us[i] * p_d <= p[tokens[i] as usize]) {
                let mut qd = vec![0.0f32; vocab];
                for &(id, v) in &q[i] {
                    qd[id as usize] = v;
                }
                let mut sum = 0.0f32;
                for c in 0..vocab {
                    let d = p[c] - qd[c];
                    if d > 0.0 {
                        sum += d;
                    }
                }
                let us2 = u_fin * sum;
                let mut acc = 0.0f32;
                for c in 0..vocab {
                    let d = p[c] - qd[c];
                    if d > 0.0 {
                        acc += d;
                        if us2 < acc {
                            return (i, c as u32);
                        }
                    }
                }
                return (i, (vocab - 1) as u32);
            }
        }
        let row = n.min(max_accept);
        let p = trunc(row);
        let mut acc = 0.0f32;
        for c in 0..vocab {
            acc += p[c];
            if u_fin < acc {
                return (row, c as u32);
            }
        }
        (row, (vocab - 1) as u32)
    }

    #[test]
    fn payload_round_trips_and_validates() {
        let p = pack(
            &[3, 5],
            &[0.25, 0.5],
            0.75,
            7,
            &[0.5, 1.0],
            &[vec![(1, 0.5), (3, 0.5)], vec![(5, 1.0)]],
        );
        assert_eq!(p.len(), dflash_accept_payload_words(2, 3));
        assert_eq!(p[2 * 2 + 1], 2, "max_accept is clamped to n");
        validate_payload(&p, 2, 8).unwrap();
        assert!(
            validate_payload(&p, 2, 5).is_err(),
            "token 5 is outside vocab 5"
        );
        let mut out = Vec::new();
        assert!(dflash_accept_pack_payload(
            &mut out,
            &[1],
            &[0.1, 0.2],
            0.5,
            1,
            &[1.0],
            &[0, 0],
            &[],
            &[]
        )
        .is_err());
        assert!(dflash_accept_pack_payload(
            &mut out,
            &[1],
            &[0.1],
            0.5,
            1,
            &[1.0],
            &[0, 2],
            &[1],
            &[1.0]
        )
        .is_err());
        // Unsorted q ids are refused by the launcher's validation.
        let bad = pack(&[0], &[0.1], 0.5, 1, &[1.0], &[vec![(4, 0.5), (2, 0.5)]]);
        assert!(validate_payload(&bad, 1, 8).is_err());
    }

    /// Decisions equal the legacy ordered loop for (almost) every uniform:
    /// the two differ only when `u` lands within f32 association error of a
    /// CDF boundary, so over thousands of windows the mismatch rate is ~0.
    #[test]
    fn reference_matches_ordered_loop_on_random_windows() {
        let vocab = 1024;
        let n = 4;
        let mut seed = 0x5eed_u64;
        let (mut checked, mut mismatched) = (0, 0);
        for keep in [0usize, 1, 3, 20, 64] {
            let (probs, tau, z) = rows(vocab, n + 1, keep, &mut seed);
            for _ in 0..800 {
                let tokens: Vec<u32> = (0..n)
                    .map(|_| (lcg(&mut seed) * vocab as f32) as u32 % vocab as u32)
                    .collect();
                let us: Vec<f32> = (0..n).map(|_| lcg(&mut seed)).collect();
                let u_fin = lcg(&mut seed);
                // q: a small support containing the proposed token (last write wins).
                let q: Vec<Vec<(u32, f32)>> = (0..n)
                    .map(|i| {
                        let mut ids: Vec<u32> = (0..6)
                            .map(|_| (lcg(&mut seed) * vocab as f32) as u32 % vocab as u32)
                            .collect();
                        ids.push(tokens[i]);
                        ids.sort_unstable();
                        ids.dedup();
                        let w = 1.0 / ids.len() as f32;
                        ids.into_iter().map(|id| (id, w)).collect()
                    })
                    .collect();
                let selq: Vec<f32> = (0..n)
                    .map(|i| q[i].iter().find(|e| e.0 == tokens[i]).unwrap().1)
                    .collect();
                let max_accept = [n, n, 2, 0][(lcg(&mut seed) * 4.0) as usize % 4];
                let p = pack(&tokens, &us, u_fin, max_accept, &selq, &q);
                let r = dflash_sampled_accept_reference(&probs, &tau, &z, &p, n, vocab);
                let (acc, bonus) = legacy(
                    &probs, &tau, &z, &tokens, &us, u_fin, max_accept, &selq, &q, vocab,
                );
                checked += 1;
                if (r.accept_len, r.bonus) != (acc, bonus) {
                    mismatched += 1;
                }
                assert!(r.accept_len <= max_accept.min(n));
                assert_eq!(r.rejected, r.accept_len < max_accept.min(n));
            }
        }
        assert!(
            mismatched * 200 <= checked,
            "{mismatched}/{checked} windows differ from the ordered loop"
        );
    }

    /// Point-mass q (PLD): rejecting `t` draws from p with `t` removed.
    #[test]
    fn point_mass_residual_never_returns_the_rejected_token() {
        let vocab = 600;
        let mut seed = 77;
        let (probs, tau, z) = rows(vocab, 2, 20, &mut seed);
        for k in 0..400u32 {
            let t = (k * 7) % vocab as u32;
            // u = 0.999 against p_d = 1: rejects unless p(t) is the near-whole row.
            let p = pack(&[t], &[0.999], lcg(&mut seed), 1, &[1.0], &[vec![(t, 1.0)]]);
            let r = dflash_sampled_accept_reference(&probs, &tau, &z, &p, 1, vocab);
            if r.rejected {
                assert_ne!(r.bonus, t, "k={k}");
                assert_eq!(r.fallback, 0);
                let row = &probs[..vocab];
                assert!(
                    row[r.bonus as usize] >= tau[0],
                    "residual leaves the kept set"
                );
            }
        }
    }

    #[test]
    fn degenerate_residual_returns_the_target_argmax() {
        // p = q everywhere on a 1-token support: u * 1 > p(t) = 1 is impossible,
        // so force the rejection with q(t) = 2 and the residual sum <= 0.
        let vocab = 16;
        let mut probs = vec![0.0f32; 2 * vocab];
        probs[3] = 1.0;
        probs[vocab + 5] = 1.0;
        let (tau, z) = (vec![1.0f32; 2], vec![1.0f32; 2]);
        let p = pack(&[3], &[0.9], 0.3, 1, &[2.0], &[vec![(3, 2.0)]]);
        let r = dflash_sampled_accept_reference(&probs, &tau, &z, &p, 1, vocab);
        // u * 2 = 1.8 > 1 rejects; residual 1 - 2 < 0 elsewhere 0: degenerate.
        assert_eq!(
            r,
            DflashDeviceDecision {
                accept_len: 0,
                bonus: 3,
                rejected: true,
                fallback: 1
            }
        );
    }

    #[test]
    fn bonus_samples_the_kept_set_only_and_budget_bounds_accept() {
        let vocab = 2000;
        let mut seed = 5;
        let (probs, tau, z) = rows(vocab, 4, 10, &mut seed);
        for k in 0..500 {
            let u_fin = lcg(&mut seed);
            // Every proposal accepted (u = 0, q = 1) but only 2 rows allowed.
            let p = pack(
                &[1, 2, 3],
                &[0.0; 3],
                u_fin,
                2,
                &[1.0; 3],
                &[vec![(1, 1.0)], vec![(2, 1.0)], vec![(3, 1.0)]],
            );
            let r = dflash_sampled_accept_reference(&probs, &tau, &z, &p, 3, vocab);
            // Token probabilities are > 0 only if kept; u = 0 accepts anything with p_t >= 0.
            assert_eq!((r.accept_len, r.rejected), (2, false), "k={k}");
            assert!(
                probs[2 * vocab + r.bonus as usize] >= tau[2],
                "bonus outside kept set"
            );
        }
    }

    /// Exact output law: x ~ q, accept w.p. min(1, p/q), else residual — the
    /// committed token is distributed as the truncated target row.
    #[test]
    fn committed_token_law_equals_the_target_row() {
        let vocab = 300;
        let mut seed = 99;
        let (probs, tau, z) = rows(vocab, 2, 8, &mut seed);
        let inv = 1.0f32 / z[0].max(FLT_MIN);
        let target: Vec<f32> = probs[..vocab]
            .iter()
            .map(|&p| eff(p, tau[0], inv))
            .collect();
        // q: uniform over 12 ids chosen to overlap and miss the nucleus.
        let mut qids: Vec<u32> = (0..12).map(|i| (i * 23 + 4) % vocab as u32).collect();
        let top = target
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0 as u32;
        qids.push(top);
        qids.sort_unstable();
        qids.dedup();
        let qv = 1.0 / qids.len() as f32;
        let trials = 200_000;
        let mut counts = vec![0u32; vocab];
        for _ in 0..trials {
            let x = qids[(lcg(&mut seed) * qids.len() as f32) as usize % qids.len()];
            let q: Vec<(u32, f32)> = qids.iter().map(|&i| (i, qv)).collect();
            let p = pack(&[x], &[lcg(&mut seed)], lcg(&mut seed), 1, &[qv], &[q]);
            let r = dflash_sampled_accept_reference(&probs, &tau, &z, &p, 1, vocab);
            let tok = if r.rejected { r.bonus } else { x };
            counts[tok as usize] += 1;
        }
        for c in 0..vocab {
            let e = target[c] as f64 * trials as f64;
            let sd = (e * (1.0 - target[c] as f64)).sqrt();
            assert!(
                (counts[c] as f64 - e).abs() <= 6.0 * sd + 3.0,
                "token {c}: {} draws, expected {e:.1} (sd {sd:.1})",
                counts[c]
            );
        }
    }

    #[test]
    fn full_support_rows_need_no_compaction() {
        // tau = 0 keeps the whole row (identity truncation): the bonus walks all
        // 4096 ids and never leaves the vocab.
        let vocab = 4096;
        let mut seed = 11;
        let (probs, tau, z) = rows(vocab, 2, 0, &mut seed);
        assert_eq!(tau[0], 0.0);
        let mut seen_hi = false;
        for _ in 0..300 {
            let p = pack(&[7], &[0.0], lcg(&mut seed), 1, &[1.0], &[vec![(7, 1.0)]]);
            let r = dflash_sampled_accept_reference(&probs, &tau, &z, &p, 1, vocab);
            assert!(!r.rejected && r.accept_len == 1 && (r.bonus as usize) < vocab);
            seen_hi |= r.bonus as usize > vocab / 2;
        }
        assert!(seen_hi, "draws must cover the whole vocabulary");
    }
}
