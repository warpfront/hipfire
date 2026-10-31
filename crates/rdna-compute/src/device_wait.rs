// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Host side of the device-clock bound on flag spin waits.
//!
//! Some kernels spin on a flag that another workgroup, GPU or the host
//! writes: the TP graph barriers, the gfx1100 RMSNorm wavegrid rendezvous and
//! the `hc_compute_control` `rsqrt_once` handoff. Each such wait
//! (`kernels/src/device_wait_deadline.h`) stops after
//! [`DEVICE_WAIT_TIMEOUT_MS`] on the GPU realtime counter and records its site
//! in this device's host-mapped [`DeviceWaitBlock`]. A stalled producer then
//! costs one bounded step instead of a wave that spins until process teardown.
//!
//! The waiting kernel's output after a timeout is garbage. The host must check
//! the block after the step with [`crate::Gpu::take_device_wait_error`]. The
//! `Gpu::download_*` reads, the batch sampler's D2H and the llama prefill
//! sample read check it automatically. A raw `hip.memcpy_dtoh` of a step's
//! result does not: such a reader must call the check itself, or a timeout
//! surfaces only at the next bounded-wait launch. The check turns a recorded
//! timeout into a `HipError`, so the request fails. Reading the block clears it.
//!
//! The block is allocated outside graph capture
//! ([`crate::Gpu::ensure_device_wait_block`]); a bounded-wait launch under
//! capture without one fails closed.

use hip_bridge::{HipError, HipResult, HipRuntime};
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{fence, Ordering};

/// Device-side bound on one wait. It is far above any healthy barrier or
/// rendezvous (microseconds) and well below the ~2.1 s MES hang-detection
/// window, after which a hung queue escalates to a GPU reset.
pub const DEVICE_WAIT_TIMEOUT_MS: u64 = 500;

/// `hipDeviceAttributeWallClockRate` (`hipDeviceAttributeAmdSpecificBegin` +
/// 17): the rate in kHz of the counter behind `wall_clock64`.
const HIP_DEVICE_ATTRIBUTE_WALL_CLOCK_RATE: i32 = 10_017;

/// Host-mapped error block that the device writes; mirrors
/// `struct hf_device_wait_block` in `device_wait_deadline.h`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceWaitBlock {
    /// 0, or the [`DeviceWaitSite`] code of a wait that stopped on its deadline.
    pub code: u32,
    /// Site detail: the barrier epoch/generation, or the waiting workgroup.
    pub detail: u32,
    /// Deadline in realtime-counter ticks. Written once at allocation.
    pub timeout_ticks: u64,
}

/// The bounded wait that recorded a timeout. The codes match
/// `HF_WAIT_SITE_*` in `device_wait_deadline.h`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum DeviceWaitSite {
    TpSignalWait2Gfx1201 = 1,
    TpSignalWait3Gfx1201 = 2,
    TpSignalBarrierGfx1100 = 3,
    TpSignalReduceGfx1100 = 4,
    RmsnormWavegridReady = 5,
    RmsnormAwqWavegridArrivals = 6,
    RmsnormAwqWavegridReady = 7,
    HcControlRsqrtOnce = 8,
    TpPeerPushGfx1201 = 9,
    TpPeerPushNGfx1201 = 10,
    EpBoundaryWait = 11,
    /// Written by the host ([`DeviceWaitPoison`]), never by a kernel.
    EpBoundaryPoison = 12,
}

impl DeviceWaitSite {
    pub const ALL: [Self; 12] = [
        Self::TpSignalWait2Gfx1201,
        Self::TpSignalWait3Gfx1201,
        Self::TpSignalBarrierGfx1100,
        Self::TpSignalReduceGfx1100,
        Self::RmsnormWavegridReady,
        Self::RmsnormAwqWavegridArrivals,
        Self::RmsnormAwqWavegridReady,
        Self::HcControlRsqrtOnce,
        Self::TpPeerPushGfx1201,
        Self::TpPeerPushNGfx1201,
        Self::EpBoundaryWait,
        Self::EpBoundaryPoison,
    ];

    pub fn from_code(code: u32) -> Option<Self> {
        Self::ALL.into_iter().find(|site| *site as u32 == code)
    }

    /// The `HF_WAIT_SITE_` suffix of the device-side define.
    pub fn header_name(self) -> &'static str {
        match self {
            Self::TpSignalWait2Gfx1201 => "TP_SIGNAL_WAIT2_GFX1201",
            Self::TpSignalWait3Gfx1201 => "TP_SIGNAL_WAIT3_GFX1201",
            Self::TpSignalBarrierGfx1100 => "TP_SIGNAL_BARRIER_GFX1100",
            Self::TpSignalReduceGfx1100 => "TP_SIGNAL_REDUCE_GFX1100",
            Self::RmsnormWavegridReady => "RMSNORM_WAVEGRID_READY",
            Self::RmsnormAwqWavegridArrivals => "RMSNORM_AWQ_WAVEGRID_ARRIVALS",
            Self::RmsnormAwqWavegridReady => "RMSNORM_AWQ_WAVEGRID_READY",
            Self::HcControlRsqrtOnce => "HC_CONTROL_RSQRT_ONCE",
            Self::TpPeerPushGfx1201 => "TP_PEER_PUSH_GFX1201",
            Self::TpPeerPushNGfx1201 => "TP_PEER_PUSHN_GFX1201",
            Self::EpBoundaryWait => "EP_BOUNDARY_WAIT",
            Self::EpBoundaryPoison => "EP_BOUNDARY_POISON",
        }
    }

    /// Kernel symbol that owns the wait.
    pub fn kernel(self) -> &'static str {
        match self {
            Self::TpSignalWait2Gfx1201 => "tp_graph_signal_wait2_gfx1201",
            Self::TpSignalWait3Gfx1201 => "tp_graph_signal_wait3_gfx1201",
            Self::TpSignalBarrierGfx1100 => "tp_graph_signal_barrier_gfx1100",
            Self::TpSignalReduceGfx1100 => "tp_graph_signal_reduce_gfx1100",
            Self::RmsnormWavegridReady => "fused_rmsnorm_mq_rotate_wavegrid",
            Self::RmsnormAwqWavegridArrivals | Self::RmsnormAwqWavegridReady => {
                "fused_rmsnorm_mq_rotate_awq_wavegrid"
            }
            Self::HcControlRsqrtOnce => "hc_compute_control_vec4_finalize (rsqrt_once)",
            Self::TpPeerPushGfx1201 => "tp2_peer_push_reduce_gfx1201",
            Self::TpPeerPushNGfx1201 => "tp_peer_pushn_reduce_bf16_gfx1201",
            Self::EpBoundaryWait | Self::EpBoundaryPoison => "ep_boundary_signal_wait",
        }
    }

    /// What never arrived.
    fn awaited(self) -> &'static str {
        match self {
            Self::TpSignalWait2Gfx1201
            | Self::TpSignalWait3Gfx1201
            | Self::TpSignalBarrierGfx1100
            | Self::TpSignalReduceGfx1100 => "a peer TP rank's barrier signal",
            Self::TpPeerPushGfx1201 | Self::TpPeerPushNGfx1201 => {
                "a peer TP rank's tagged partial granules"
            }
            Self::RmsnormWavegridReady
            | Self::RmsnormAwqWavegridReady
            | Self::RmsnormAwqWavegridArrivals => {
                "the wavegrid rendezvous (workgroups not co-resident)"
            }
            Self::HcControlRsqrtOnce => "control row 0's rsqrt",
            Self::EpBoundaryWait | Self::EpBoundaryPoison => "every EP rank's boundary epoch",
        }
    }

    fn detail_label(self) -> &'static str {
        match self {
            Self::TpSignalWait2Gfx1201 | Self::TpSignalWait3Gfx1201 | Self::EpBoundaryWait => {
                "epoch"
            }
            Self::EpBoundaryPoison => "poison",
            Self::TpSignalBarrierGfx1100
            | Self::TpSignalReduceGfx1100
            | Self::TpPeerPushGfx1201
            | Self::TpPeerPushNGfx1201 => "generation",
            Self::RmsnormAwqWavegridArrivals => "arrival",
            Self::RmsnormWavegridReady
            | Self::RmsnormAwqWavegridReady
            | Self::HcControlRsqrtOnce => "workgroup",
        }
    }
}

/// A timeout read back from a device's [`DeviceWaitBlock`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceWaitTimeout {
    pub device_id: i32,
    pub code: u32,
    pub detail: u32,
}

impl DeviceWaitTimeout {
    pub fn site(&self) -> Option<DeviceWaitSite> {
        DeviceWaitSite::from_code(self.code)
    }

    /// The request-failing error. This is explicit rather than a
    /// `From<_> for HipError` impl: a second `From` impl would break the
    /// error-type inference of existing `?` closures.
    pub fn to_hip_error(&self) -> HipError {
        HipError::new(0, &self.to_string())
    }
}

impl fmt::Display for DeviceWaitTimeout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let device = self.device_id;
        match self.site() {
            Some(site @ DeviceWaitSite::EpBoundaryPoison) => write!(
                f,
                "EP boundary poisoned on GPU {device}: another rank failed, so {} stopped \
                 waiting ({} {}); this step's results are invalid and the request was aborted",
                site.kernel(),
                site.detail_label(),
                self.detail,
            ),
            Some(site) => write!(
                f,
                "device wait timeout on GPU {device}: {} waited {DEVICE_WAIT_TIMEOUT_MS} ms for {} \
                 ({} {}) and stopped; this step's results are invalid and the request was aborted",
                site.kernel(),
                site.awaited(),
                site.detail_label(),
                self.detail,
            ),
            None => write!(
                f,
                "device wait timeout on GPU {device}: unknown wait site code {} (detail {}); \
                 this step's results are invalid and the request was aborted",
                self.code, self.detail,
            ),
        }
    }
}

/// Realtime-counter ticks in [`DEVICE_WAIT_TIMEOUT_MS`] at `wall_clock_khz`.
pub fn timeout_ticks(wall_clock_khz: u64) -> u64 {
    wall_clock_khz.saturating_mul(DEVICE_WAIT_TIMEOUT_MS)
}

/// Read the block and clear a recorded timeout.
///
/// # Safety
/// `block` must point to a live, aligned `DeviceWaitBlock`. The device may
/// write it concurrently. A store that lands after the clear stays set for
/// the next check; one that lands between the read and the clear is lost,
/// which only drops a second code while a first is already being reported.
pub unsafe fn take_block(block: *mut DeviceWaitBlock, device_id: i32) -> Option<DeviceWaitTimeout> {
    use std::ptr::{addr_of, addr_of_mut, read_volatile, write_volatile};
    // SAFETY: caller guarantees a live block; the device writes it with
    // system-scope stores, so every access is volatile.
    let code = unsafe { read_volatile(addr_of!((*block).code)) };
    if code == 0 {
        return None;
    }
    // The device publishes `detail` before `code` (release).
    fence(Ordering::Acquire);
    // SAFETY: as above.
    let detail = unsafe { read_volatile(addr_of!((*block).detail)) };
    // SAFETY: as above.
    unsafe {
        write_volatile(addr_of_mut!((*block).detail), 0);
        write_volatile(addr_of_mut!((*block).code), 0);
    }
    Some(DeviceWaitTimeout {
        device_id,
        code,
        detail,
    })
}

/// One device's host-mapped [`DeviceWaitBlock`], owned by `Gpu`. The block
/// lives as long as the `Gpu`, so captured graphs and retained replay tapes
/// that baked in its device address stay valid.
#[derive(Debug)]
pub(crate) struct DeviceWaitMapping {
    /// `hipHostMalloc` pointer (host view, and what `hipHostFree` takes).
    host: usize,
    /// `hipHostGetDevicePointer` alias handed to kernels.
    device: usize,
}

impl DeviceWaitMapping {
    pub(crate) fn alloc(hip: &HipRuntime, device_id: i32) -> HipResult<Self> {
        let khz = hip.get_device_attribute(HIP_DEVICE_ATTRIBUTE_WALL_CLOCK_RATE, device_id)?;
        let khz =
            u64::try_from(khz)
                .ok()
                .filter(|&khz| khz > 0)
                .ok_or_else(|| {
                    HipError::new(
                0,
                &format!("device {device_id}: hipDeviceAttributeWallClockRate returned {khz} kHz"),
            )
                })?;
        let host = hip.host_malloc(
            std::mem::size_of::<DeviceWaitBlock>(),
            hip_bridge::HIP_HOST_MALLOC_MAPPED | hip_bridge::HIP_HOST_MALLOC_COHERENT,
        )?;
        let device = match hip.host_get_device_pointer(host, 0) {
            Ok(device) if !device.is_null() => device,
            other => {
                let _ = hip.host_free(host);
                return Err(match other {
                    Err(error) => error,
                    Ok(_) => HipError::new(
                        0,
                        &format!("device {device_id}: null device alias for the wait-error block"),
                    ),
                });
            }
        };
        // SAFETY: fresh hipHostMalloc allocation of exactly one block, which
        // is suitably aligned (page-aligned) and not yet visible to any kernel.
        unsafe {
            host.cast::<DeviceWaitBlock>().write(DeviceWaitBlock {
                code: 0,
                detail: 0,
                timeout_ticks: timeout_ticks(khz),
            });
        }
        Ok(Self {
            host: host as usize,
            device: device as usize,
        })
    }

    pub(crate) fn device_ptr(&self) -> *mut c_void {
        self.device as *mut c_void
    }

    pub(crate) fn take(&self, device_id: i32) -> Option<DeviceWaitTimeout> {
        // SAFETY: `host` is the live allocation made in `alloc`; it is freed
        // only by `free`, which consumes `self`.
        unsafe { take_block(self.host as *mut DeviceWaitBlock, device_id) }
    }

    /// Release the block. The caller must have synchronized the device.
    pub(crate) fn free(self, hip: &HipRuntime) -> HipResult<()> {
        hip.host_free(self.host as *mut c_void)
    }

    pub(crate) fn poison_handle(&self) -> DeviceWaitPoison {
        DeviceWaitPoison { host: self.host }
    }
}

/// Stops one device's bounded waits from any host thread: it stores
/// [`DeviceWaitSite::EpBoundaryPoison`] into the device's wait block, so every
/// waiter on that device stops at its next check (`hf_wait_expired`) and the
/// host's step check reports the poison. An EP mesh uses it to release ranks
/// spinning on a boundary whose peer failed on the host.
#[derive(Clone, Copy, Debug)]
pub struct DeviceWaitPoison {
    host: usize,
}

impl DeviceWaitPoison {
    /// # Safety
    /// The owning `Gpu` (and so its wait block) must still be alive.
    pub unsafe fn poison(&self, detail: u32) {
        use std::ptr::{addr_of_mut, write_volatile};
        let block = self.host as *mut DeviceWaitBlock;
        // SAFETY: the caller guarantees the block is live. A device timeout
        // racing this store is fine: any nonzero code fails the step.
        unsafe {
            write_volatile(addr_of_mut!((*block).detail), detail);
            fence(Ordering::Release);
            write_volatile(
                addr_of_mut!((*block).code),
                DeviceWaitSite::EpBoundaryPoison as u32,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_layout_matches_the_device_struct() {
        assert_eq!(std::mem::size_of::<DeviceWaitBlock>(), 16);
        assert_eq!(std::mem::offset_of!(DeviceWaitBlock, code), 0);
        assert_eq!(std::mem::offset_of!(DeviceWaitBlock, detail), 4);
        assert_eq!(std::mem::offset_of!(DeviceWaitBlock, timeout_ticks), 8);
    }

    #[test]
    fn site_codes_match_the_device_header() {
        let mut defines = Vec::new();
        for line in crate::kernels::DEVICE_WAIT_DEADLINE_H.lines() {
            let Some(rest) = line.strip_prefix("#define HF_WAIT_SITE_") else {
                continue;
            };
            let mut parts = rest.split_whitespace();
            let name = parts.next().expect("define name");
            let value = parts.next().expect("define value");
            let code: u32 = value
                .trim_end_matches('u')
                .parse()
                .expect("numeric site code");
            defines.push((name.to_string(), code));
        }
        assert_eq!(defines.len(), DeviceWaitSite::ALL.len());
        for (name, code) in defines {
            let site = DeviceWaitSite::from_code(code)
                .unwrap_or_else(|| panic!("header site {name}={code} has no Rust site"));
            assert_eq!(site.header_name(), name, "code {code}");
        }
    }

    #[test]
    fn timeout_is_500_ms_of_the_realtime_counter() {
        // RDNA3/RDNA4/gfx1151 report a 100 MHz realtime counter.
        assert_eq!(timeout_ticks(100_000), 50_000_000);
        assert_eq!(timeout_ticks(25_000), 12_500_000);
    }

    #[test]
    fn host_error_path_reports_and_clears_a_recorded_timeout() {
        // Stand-in for the host-mapped block; every access below goes through
        // `ptr`, as both the device and the host access the real block by address.
        let ptr = Box::into_raw(Box::new(DeviceWaitBlock {
            code: 0,
            detail: 0,
            timeout_ticks: timeout_ticks(100_000),
        }));

        // SAFETY: `ptr` is the live boxed block until `Box::from_raw` below.
        assert_eq!(
            unsafe { take_block(ptr, 2) },
            None,
            "clean block reports nothing"
        );

        // What `hf_wait_expired` stores when a TP4 rank's wait3 hits its deadline.
        // SAFETY: as above.
        unsafe {
            (*ptr).detail = 37;
            (*ptr).code = DeviceWaitSite::TpSignalWait3Gfx1201 as u32;
        }
        // SAFETY: as above.
        let timeout = unsafe { take_block(ptr, 2) }.expect("recorded timeout is reported");
        assert_eq!(
            timeout,
            DeviceWaitTimeout {
                device_id: 2,
                code: 2,
                detail: 37,
            }
        );
        let error = timeout.to_hip_error();
        let message = error.to_string();
        for needle in [
            "device wait timeout on GPU 2",
            "tp_graph_signal_wait3_gfx1201",
            "500 ms",
            "peer TP rank",
            "epoch 37",
            "request was aborted",
        ] {
            assert!(
                message.contains(needle),
                "{needle:?} missing from {message:?}"
            );
        }

        // SAFETY: as above.
        assert_eq!(
            unsafe { take_block(ptr, 2) },
            None,
            "a timeout is reported once"
        );
        // SAFETY: `ptr` came from `Box::into_raw` and is not used afterwards.
        let block = unsafe { Box::from_raw(ptr) };
        // Reading clears the code and detail, but not the deadline.
        assert_eq!(
            *block,
            DeviceWaitBlock {
                code: 0,
                detail: 0,
                timeout_ticks: 50_000_000,
            }
        );
    }

    #[test]
    fn every_site_and_unknown_codes_render_a_clear_message() {
        for site in DeviceWaitSite::ALL {
            let message = DeviceWaitTimeout {
                device_id: 0,
                code: site as u32,
                detail: 3,
            }
            .to_string();
            assert!(message.contains(site.kernel()), "{message}");
            assert!(
                message.contains(&format!("{} 3", site.detail_label())),
                "{message}"
            );
        }
        let unknown = DeviceWaitTimeout {
            device_id: 1,
            code: 99,
            detail: 4,
        }
        .to_string();
        assert!(unknown.contains("unknown wait site code 99"), "{unknown}");
    }
}
