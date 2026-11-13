// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Device-clock bound for spin waits on a flag that another workgroup, GPU or
// the host writes.
//
// A wait that the producer never releases (a stalled or reset peer GPU, or
// workgroups that are not co-resident) would otherwise keep its wave resident
// until process teardown. A reset on any card suspends every KFD process on
// the host, so no wait may outlive the MES hang window (~2.1 s).
//
// Each bounded wait reads the constant-rate realtime counter (`wall_clock64`:
// s_sendmsg_rtn_b64 MSG_RTN_GET_REALTIME on gfx11/gfx12, s_memrealtime on
// older targets). The first poll and the loop body match the unbounded wait,
// including the s_sleep. Every HF_WAIT_CHECK_INTERVAL polls the waiter reads
// the counter. A short wait pays only that read. Once the wait passes
// HF_WAIT_QUIET_TICKS, the waiter also reads the host-mapped error block, a
// PCIe round trip, at most once per HF_WAIT_HOST_CHECK_TICKS:
//   - The error block's code is nonzero. An earlier wait on this device timed
//     out in this step, so the waiter stops rather than spending its own full
//     deadline.
//   - The deadline (the block's timeout_ticks) has passed. The waiter records
//     its site and detail, then stops waiting.
// The caller decides what "stop" means: TP barriers exit the wave, and
// rendezvous kernels finish their epilogue so their self-restoring scratch
// stays consistent.
//
// The host maps the error block (rdna-compute `device_wait.rs`,
// `DeviceWaitBlock`), reads it after the step and fails the request. The
// writes are plain system-scope stores, not RMW atomics, so PCIe atomics are
// not required. With several timed-out waiters the last store wins, and any
// nonzero code fails the step.
//
// rdna-compute prepends this header to each kernel source that uses it. The
// runtime hipcc compile has no -I to kernels/src, so each such source wraps
// its #include in #ifndef HF_DEVICE_WAIT_DEADLINE_H.

#ifndef HF_DEVICE_WAIT_DEADLINE_H
#define HF_DEVICE_WAIT_DEADLINE_H

#include <hip/hip_runtime.h>

// Mirrors `DeviceWaitBlock` (`#[repr(C)]`, 16 bytes).
struct hf_device_wait_block {
    unsigned int code;                 // 0, or the HF_WAIT_SITE_* of a timed-out wait
    unsigned int detail;               // barrier epoch, or workgroup index
    unsigned long long timeout_ticks;  // realtime-counter ticks, host-written once
};

// Site codes. Keep in sync with `DeviceWaitSite` in device_wait.rs.
#define HF_WAIT_SITE_TP_SIGNAL_WAIT2_GFX1201 1u
#define HF_WAIT_SITE_TP_SIGNAL_WAIT3_GFX1201 2u
#define HF_WAIT_SITE_TP_SIGNAL_BARRIER_GFX1100 3u
#define HF_WAIT_SITE_TP_SIGNAL_REDUCE_GFX1100 4u
#define HF_WAIT_SITE_RMSNORM_WAVEGRID_READY 5u
#define HF_WAIT_SITE_RMSNORM_AWQ_WAVEGRID_ARRIVALS 6u
#define HF_WAIT_SITE_RMSNORM_AWQ_WAVEGRID_READY 7u
#define HF_WAIT_SITE_HC_CONTROL_RSQRT_ONCE 8u
#define HF_WAIT_SITE_TP_PEER_PUSH_GFX1201 9u
#define HF_WAIT_SITE_TP_PEER_PUSHN_GFX1201 10u
#define HF_WAIT_SITE_AR_HC_WRITE_GFX1201 11u

// Polls between counter reads. A wait that ends within this many polls never
// reads the counter.
#define HF_WAIT_CHECK_INTERVAL 16u

// Elapsed ticks before a waiter first reads the host-mapped error block: 1 ms
// at the 100 MHz realtime counter of RDNA3/RDNA4. Below this, a check costs one
// counter read and no PCIe round trip. Healthy waits do pass 100 us: an EP2
// decode all-reduce waits out the peer's routed-expert skew, 100-630 us on
// some layers of most steps (rocprof j8299: p99.9 281-511 us, 3 of ~27k waits
// over 1 ms). The previous 100 us quiet period with a read at EVERY check
// (every 16 polls) made each waiting wave issue two host reads per
// microsecond; with 256-960 waves polling, the reads queued at the root
// complex and a waiter noticed the peer's data 30-120 us late instead of
// ~8-15 us (j8299 served: the waiting rank left the all-reduce 40 / 77 us after
// the peer's arrival on waits over 100 us, 8.3 / 8.8 us below, story / code C1;
// lab bench j8304). Even one read per wave per 100 us still cost 20-50 us when
// the arrival met the synchronized burst of every wave's read.
#define HF_WAIT_QUIET_TICKS 100000ull

// Ticks between two reads of the host-mapped error block by one waiter after
// HF_WAIT_QUIET_TICKS (1 ms). Stopping on a timeout recorded by another wait
// takes at most this long; the 500 ms deadline is checked at the same cadence.
#define HF_WAIT_HOST_CHECK_TICKS 100000ull

// True when the waiter must stop: either another wait already recorded a
// timeout on this device, or this wait has passed its deadline (recorded here).
__device__ __forceinline__ bool hf_wait_expired(
    hf_device_wait_block *block,
    unsigned long long elapsed,
    unsigned int site,
    unsigned int detail) {
    if (elapsed < HF_WAIT_QUIET_TICKS) {
        return false;
    }
    if (__hip_atomic_load(&block->code, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) != 0u) {
        return true;
    }
    if (elapsed < __hip_atomic_load(
                      &block->timeout_ticks, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)) {
        return false;
    }
    __hip_atomic_store(&block->detail, detail, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    __hip_atomic_store(&block->code, site, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    return true;
}

// Spin until `ready()` holds, with s_sleep(1) between polls, exactly like
// the unbounded `while (!ready()) s_sleep(1);`. Returns true when the wait
// was satisfied. Returns false when it stopped on the deadline or because
// another wait had already timed out.
//
// The inner do/while is the unbounded loop plus one decrement folded into
// its exit test. Other shapes of this loop (a modulo counter, a nested for
// with an early return) compiled to extra SALU per poll and slowed the
// gfx1100 rendezvous kernels by 1-3% on gfx1201. The deadline runs from the
// first counter read, HF_WAIT_CHECK_INTERVAL polls in. The polls before it
// add microseconds to a 500 ms bound. The host block is read first at
// HF_WAIT_QUIET_TICKS, then every HF_WAIT_HOST_CHECK_TICKS.
template <typename Ready>
__device__ __forceinline__ bool hf_bounded_wait(
    Ready ready,
    hf_device_wait_block *block,
    unsigned int site,
    unsigned int detail) {
    if (ready()) {
        return true;
    }
    unsigned long long start = 0ull;
    unsigned long long next_host = HF_WAIT_QUIET_TICKS;
    for (bool first_check = true;; first_check = false) {
        unsigned int budget = HF_WAIT_CHECK_INTERVAL;
        bool done;
        do {
            __builtin_amdgcn_s_sleep(1);
            done = ready();
        } while (!done && --budget != 0u);
        if (done) {
            return true;
        }
        const unsigned long long now = wall_clock64();
        if (first_check) {
            start = now;
        } else if (now - start >= next_host) {
            if (hf_wait_expired(block, now - start, site, detail)) {
                return false;
            }
            next_host = now - start + HF_WAIT_HOST_CHECK_TICKS;
        }
    }
}

#endif  // HF_DEVICE_WAIT_DEADLINE_H
