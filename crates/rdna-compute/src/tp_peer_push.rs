// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1201 TP2 residual reduction through tagged peer-VRAM granules
//! (`kernels/src/tp2_peer_push.gfx1201.hip`). The runtime owns the
//! fine-grained mailboxes and generation counters; this module only resolves
//! the kernel and launches it.

use crate::dispatch::Gpu;
use crate::kernels;
use hip_bridge::{DeviceBuffer, HipError, HipResult, KernargBlob};
use std::ffi::c_void;

/// Lanes of the single workgroup; the kernel is written for exactly this.
pub const TP_PEER_PUSH_LANES: u32 = 512;
/// Largest reduction (floats) the kernel serves: 8 granules of 2 floats per lane.
pub const TP_PEER_PUSH_MAX_COUNT: usize = 8 * TP_PEER_PUSH_LANES as usize * 2;
/// Bytes per tagged granule (two floats and the generation tag twice).
pub const TP_PEER_PUSH_GRANULE_BYTES: usize = 16;

const KERNEL: &str = "tp2_peer_push_reduce_gfx1201";

impl Gpu {
    /// Resolve the peer-push kernel before graph capture (JIT is not
    /// capture-safe).
    pub fn ensure_tp_peer_push_gfx1201_kernel(&mut self) -> HipResult<()> {
        self.bind_thread()?;
        if self.arch != "gfx1201" {
            return Err(HipError::new(
                0,
                &format!(
                    "tp2_peer_push_reduce_gfx1201 requires gfx1201, got {}",
                    self.arch
                ),
            ));
        }
        self.ensure_kernel(KERNEL, kernels::TP_PEER_PUSH_GFX1201_SRC, KERNEL)
    }

    /// Push this rank's f32 partial into the peer's mailbox bank, wait (bounded)
    /// for the peer's granules in this rank's own mailbox, and add
    /// `x + (p0 + p1)` in rank order into `x`. `peer_mailbox` is the PEER's
    /// mailbox (peer access enabled), `local_mailbox` this rank's own; each
    /// holds two banks `bank_stride` bytes apart. `generation` is this rank's
    /// device-resident generation word. `count` must be even and at most
    /// [`TP_PEER_PUSH_MAX_COUNT`].
    #[allow(clippy::too_many_arguments)]
    pub fn tp_peer_push_reduce_gfx1201(
        &mut self,
        x: &DeviceBuffer,
        local_partial: &DeviceBuffer,
        peer_mailbox: *mut c_void,
        local_mailbox: *const c_void,
        bank_stride: usize,
        generation: &DeviceBuffer,
        local_is_first: bool,
        count: usize,
    ) -> HipResult<()> {
        self.ensure_tp_peer_push_gfx1201_kernel()?;
        if count == 0 || !count.is_multiple_of(2) || count > TP_PEER_PUSH_MAX_COUNT {
            return Err(HipError::new(
                0,
                &format!("tp2_peer_push_reduce_gfx1201: unsupported count {count}"),
            ));
        }
        if bank_stride < count / 2 * TP_PEER_PUSH_GRANULE_BYTES || !bank_stride.is_multiple_of(256)
        {
            return Err(HipError::new(
                0,
                &format!(
                    "tp2_peer_push_reduce_gfx1201: bank stride {bank_stride} cannot hold {count} floats"
                ),
            ));
        }
        let x_ptr = x.as_ptr();
        let partial_ptr = local_partial.as_ptr();
        let gen_ptr = generation.as_ptr();
        let stride_arg = u32::try_from(bank_stride)
            .map_err(|_| HipError::new(0, "tp2_peer_push_reduce_gfx1201: stride exceeds u32"))?;
        let first_arg = i32::from(local_is_first);
        let count_arg = count as i32;
        let wait_block = self.device_wait_block()?;
        let mut params: [*mut c_void; 9] = [
            &x_ptr as *const _ as *mut c_void,
            &partial_ptr as *const _ as *mut c_void,
            &peer_mailbox as *const _ as *mut c_void,
            &local_mailbox as *const _ as *mut c_void,
            &stride_arg as *const _ as *mut c_void,
            &gen_ptr as *const _ as *mut c_void,
            &wait_block as *const _ as *mut c_void,
            &first_arg as *const _ as *mut c_void,
            &count_arg as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            KERNEL,
            [1, 1, 1],
            [TP_PEER_PUSH_LANES, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(x_ptr);
                b.push_ptr(partial_ptr);
                b.push_ptr(peer_mailbox);
                b.push_ptr(local_mailbox);
                b.push_u32(stride_arg);
                b.push_ptr(gen_ptr);
                b.push_ptr(wait_block);
                b.push_i32(first_arg);
                b.push_i32(count_arg);
                b
            },
        )
    }
}

/// Granules per tagged slot of the bf16 TP3/TP4 kernel: four bf16 values per
/// 16-byte granule.
pub const TP_PEER_PUSHN_MAX_COUNT: usize = 4 * TP_PEER_PUSH_LANES as usize * 4;

const KERNEL_N: &str = "tp_peer_pushn_reduce_bf16_gfx1201";

impl Gpu {
    /// Resolve the TP3/TP4 bf16 peer-push kernel before graph capture.
    pub fn ensure_tp_peer_pushn_gfx1201_kernel(&mut self) -> HipResult<()> {
        self.bind_thread()?;
        if self.arch != "gfx1201" {
            return Err(HipError::new(
                0,
                &format!(
                    "tp_peer_pushn_reduce_bf16_gfx1201 requires gfx1201, got {}",
                    self.arch
                ),
            ));
        }
        self.ensure_kernel(KERNEL_N, kernels::TP_PEER_PUSHN_GFX1201_SRC, KERNEL_N)
    }

    /// bf16-exchange all-to-all peer-push reduction for 3 or 4 ranks: pack this
    /// rank's f32 partial to bf16, push it to slot `rank` of every peer's
    /// mailbox bank, wait (bounded) for every peer's slot in this rank's own
    /// mailbox, and add `x + (r(p0) + r(p1) + ...)` in rank order into `x`.
    /// `mailboxes[q]` is rank q's mailbox (peer access enabled; entries at or
    /// beyond `nranks` unused). Each mailbox holds two banks `bank_stride`
    /// bytes apart, each bank `nranks` slots `slot_stride` bytes apart.
    /// `count` must be a multiple of 4 and at most [`TP_PEER_PUSHN_MAX_COUNT`].
    #[allow(clippy::too_many_arguments)]
    pub fn tp_peer_pushn_reduce_bf16_gfx1201(
        &mut self,
        x: &DeviceBuffer,
        local_partial: &DeviceBuffer,
        mailboxes: [*mut c_void; 4],
        slot_stride: usize,
        bank_stride: usize,
        generation: &DeviceBuffer,
        rank: usize,
        nranks: usize,
        count: usize,
    ) -> HipResult<()> {
        self.ensure_tp_peer_pushn_gfx1201_kernel()?;
        if !(3..=4).contains(&nranks)
            || rank >= nranks
            || count == 0
            || !count.is_multiple_of(4)
            || count > TP_PEER_PUSHN_MAX_COUNT
        {
            return Err(HipError::new(
                0,
                &format!(
                    "tp_peer_pushn_reduce_bf16_gfx1201: unsupported rank {rank} of {nranks}, count {count}"
                ),
            ));
        }
        if slot_stride < count / 4 * TP_PEER_PUSH_GRANULE_BYTES
            || !slot_stride.is_multiple_of(256)
            || bank_stride < nranks * slot_stride
        {
            return Err(HipError::new(
                0,
                &format!(
                    "tp_peer_pushn_reduce_bf16_gfx1201: strides {slot_stride}/{bank_stride} cannot hold {count} floats for {nranks} ranks"
                ),
            ));
        }
        let x_ptr = x.as_ptr();
        let partial_ptr = local_partial.as_ptr();
        let gen_ptr = generation.as_ptr();
        let slot_arg = u32::try_from(slot_stride).map_err(|_| {
            HipError::new(0, "tp_peer_pushn_reduce_bf16_gfx1201: stride exceeds u32")
        })?;
        let bank_arg = u32::try_from(bank_stride).map_err(|_| {
            HipError::new(0, "tp_peer_pushn_reduce_bf16_gfx1201: stride exceeds u32")
        })?;
        let rank_arg = rank as i32;
        let nranks_arg = nranks as i32;
        let count_arg = count as i32;
        let wait_block = self.device_wait_block()?;
        let mut params: [*mut c_void; 13] = [
            &x_ptr as *const _ as *mut c_void,
            &partial_ptr as *const _ as *mut c_void,
            &mailboxes[0] as *const _ as *mut c_void,
            &mailboxes[1] as *const _ as *mut c_void,
            &mailboxes[2] as *const _ as *mut c_void,
            &mailboxes[3] as *const _ as *mut c_void,
            &slot_arg as *const _ as *mut c_void,
            &bank_arg as *const _ as *mut c_void,
            &gen_ptr as *const _ as *mut c_void,
            &wait_block as *const _ as *mut c_void,
            &rank_arg as *const _ as *mut c_void,
            &nranks_arg as *const _ as *mut c_void,
            &count_arg as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            KERNEL_N,
            [1, 1, 1],
            [TP_PEER_PUSH_LANES, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(x_ptr);
                b.push_ptr(partial_ptr);
                for m in mailboxes {
                    b.push_ptr(m);
                }
                b.push_u32(slot_arg);
                b.push_u32(bank_arg);
                b.push_ptr(gen_ptr);
                b.push_ptr(wait_block);
                b.push_i32(rank_arg);
                b.push_i32(nranks_arg);
                b.push_i32(count_arg);
                b
            },
        )
    }
}

/// Longest injected stall: well inside the ~2.1 s window in which a stuck
/// dispatch would suspend every process on the host.
pub const TP_PEER_PUSH_FAULT_MAX_MS: u32 = 1500;

impl Gpu {
    /// Developer-only fault injection (`HIPFIRE_TP_PEER_PUSH_FAULT`): enqueue a
    /// single-lane spin of `ms` milliseconds (at most
    /// [`TP_PEER_PUSH_FAULT_MAX_MS`]) on this rank's stream, so the peers of the
    /// next replay wait out their bounded deadline. Resolve with
    /// [`Self::ensure_tp_peer_push_fault_kernel`] before graph capture.
    pub fn ensure_tp_peer_push_fault_kernel(&mut self) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            FAULT_MODULE,
            kernels::TP_PEER_PUSHN_GFX1201_SRC,
            FAULT_KERNEL,
        )
    }

    pub fn tp_peer_push_fault_spin_gfx1201(&mut self, ms: u32) -> HipResult<()> {
        self.ensure_tp_peer_push_fault_kernel()?;
        let ticks = ms.min(TP_PEER_PUSH_FAULT_MAX_MS) * 100_000;
        let mut params: [*mut c_void; 1] = [&ticks as *const _ as *mut c_void];
        self.launch_maybe_blob(FAULT_KERNEL, [1, 1, 1], [64, 1, 1], 0, &mut params, || {
            let mut b = KernargBlob::new();
            b.push_u32(ticks);
            b
        })
    }
}

const FAULT_MODULE: &str = "tp_peer_push_fault_spin_gfx1201";
const FAULT_KERNEL: &str = "tp_peer_push_fault_spin_gfx1201";
