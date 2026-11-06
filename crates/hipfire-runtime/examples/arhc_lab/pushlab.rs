// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `ar_hc_write_bench --push-lab`: prefill push-rate and push/apply overlap probes (lab only).
//!
//! One kernel, both ranks at once (both link directions loaded, as in an RS/AG launch), every
//! series spin-gated like `--breakdown`:
//!  * push width / wave sweep: `n` dense 8-byte granules (one 256-row RS launch = 163840) into
//!    the peer's fine-grained sink with system-scope b64 stores (one granule per store) or b128
//!    stores (two granules per store), at several (blocks, lanes) shapes;
//!  * overlap: the same push from half the lanes and the production HC apply (`arhc_load` +
//!    `arhc_apply`, BF16 streams, 256 rows, local `m`) from the other half, with the halves split
//!    inside every block (waves) or across blocks; push only / apply only / both.
//!    `both ~= max(push, apply)` means a role-split RS/AG kernel can hide its apply behind the
//!    wire; `both ~= push + apply` means the halves serialize in the memory pipeline.

use hip_bridge::{DeviceBuffer, KernargBlob};
use rdna_compute::ar_hc_write::AR_HC_HIDDEN;
use rdna_compute::Gpu;
use std::ffi::c_void;
use std::sync::{Arc, Barrier};

pub const PUSH_SRC: &str = r#"
// ---- ar_hc_write_bench --push-lab kernels (NOT product kernels) ----
// mode bit 0: push, bit 1: apply. intra: lanes [0, T/2) push and [T/2, T) apply in every block;
// else blocks [0, push_blocks) push and the rest apply. width 8: b64 per granule, 16: b128 per
// two granules.
extern "C" __global__ __launch_bounds__(512) void plab_kernel(
    char *peer, unsigned n_push, int width, int push_blocks, int mode, int intra,
    const float *__restrict__ part, unsigned short *__restrict__ streams, const float *__restrict__ gates,
    int apply_granules, unsigned tag) {
    const int T = (int)blockDim.x, tid = (int)threadIdx.x, b = (int)blockIdx.x, nb = (int)gridDim.x;
    bool pusher;
    int id, P, A;
    if (intra) {
        pusher = tid < T / 2;
        id = b * (T / 2) + (pusher ? tid : tid - T / 2);
        P = nb * (T / 2);
        A = nb * (T / 2);
    } else {
        pusher = b < push_blocks;
        id = (pusher ? b : b - push_blocks) * T + tid;
        P = push_blocks * T;
        A = (nb - push_blocks) * T;
    }
    if (pusher) {
        if ((mode & 1) == 0) return;
        const unsigned v = tag | 1u;
        if (width == 16) {
            arhc_u4 *p = reinterpret_cast<arhc_u4 *>(peer);
            for (unsigned j = id; j < n_push / 2; j += P) arhc_st128_sys(p + j, arhc_u4{v, v, v, v});
        } else {
            arhc_u2 *p = reinterpret_cast<arhc_u2 *>(peer);
            for (unsigned j = id; j < n_push; j += P) arhc_st64_sys(p + j, arhc_u2{v, v});
        }
    } else {
        if ((mode & 2) == 0) return;
        for (int j = id; j < apply_granules; j += A) {
            const float4 a = *reinterpret_cast<const float4 *>(part + (long long)j * 4);
            const float m[4] = {arhc_bf16r(a.x), arhc_bf16r(a.y), arhc_bf16r(a.z), arhc_bf16r(a.w)};
            const long long e0 = (long long)j * 4;
            ArhcPre<false> pre;
            arhc_load<false>(streams, gates, (int)(e0 / AR_HC_HIDDEN), (int)(e0 % AR_HC_HIDDEN), pre);
            arhc_apply<false>(streams, (int)(e0 / AR_HC_HIDDEN), (int)(e0 % AR_HC_HIDDEN), pre, m);
        }
    }
}

extern "C" __global__ void plab_spin(unsigned long long ticks) {
    const unsigned long long t0 = wall_clock64();
    while (wall_clock64() - t0 < ticks) __builtin_amdgcn_s_sleep(8);
}
"#;

const ROWS: usize = 256;
const SINK_BYTES: usize = 8 << 20;

pub struct PushBufs {
    pub sink: DeviceBuffer,
}

/// The peer-written sink of one rank. Call BEFORE peer access is enabled.
pub fn alloc(gpu: &mut Gpu) -> Result<PushBufs, String> {
    let sink = gpu
        .hip
        .malloc_peer_finegrained(SINK_BYTES)
        .map_err(|e| e.to_string())?;
    gpu.hip
        .memset(&sink, 0, SINK_BYTES)
        .map_err(|e| e.to_string())?;
    Ok(PushBufs { sink })
}

/// (label, blocks, lanes, width, push_blocks, mode, intra, push granules)
type Cfg = (String, u32, u32, i32, i32, i32, i32, u32);

fn configs() -> Vec<Cfg> {
    let g = (ROWS * AR_HC_HIDDEN / 4) as u32; // 163840 granules = one 256-row RS launch
    let mut v: Vec<Cfg> = Vec::new();
    for (b, t) in [
        (54u32, 512u32),
        (64, 512),
        (64, 256),
        (32, 512),
        (128, 512),
        (256, 256),
        (16, 512),
    ] {
        for w in [8, 16] {
            v.push((format!("push w{w} {b}x{t}"), b, t, w, b as i32, 1, 0, g));
        }
    }
    for w in [8, 16] {
        v.push((
            format!("push w{w} 64x512 4x bytes"),
            64,
            512,
            w,
            64,
            1,
            0,
            4 * g,
        ));
    }
    for (name, intra, pb) in [
        ("intra 64x512 half waves", 1, 0),
        ("blocks 32+32 x512", 0, 32),
        ("blocks 48+16 x512", 0, 48),
    ] {
        for (mode, m) in [(1, "push"), (2, "apply"), (3, "both")] {
            v.push((format!("ovl {name} {m}"), 64, 512, 8, pb, mode, intra, g));
        }
    }
    v
}

/// Run every config on one rank (both ranks call this with identical sequences).
pub fn run(
    gpu: &mut Gpu,
    peer_sink: usize,
    barrier: &Arc<Barrier>,
    iters: usize,
) -> Result<Vec<(String, f64)>, String> {
    let e = |e: hip_bridge::HipError| e.to_string();
    let src = format!(
        "{}{}",
        rdna_compute::kernels::AR_HC_WRITE_GFX1201_SRC,
        PUSH_SRC
    );
    for s in ["plab_kernel", "plab_spin"] {
        gpu.ensure_kernel_public("arhc_pushlab", &src, s)
            .map_err(e)?;
    }
    let n = ROWS * AR_HC_HIDDEN;
    let part = gpu.hip.malloc(n * 4).map_err(e)?;
    let streams = gpu.hip.malloc(n * 4 * 2).map_err(e)?;
    let gates = gpu.hip.malloc(ROWS * 16).map_err(e)?;
    let hp: Vec<u8> = (0..n)
        .flat_map(|i| (((i % 97) as f32 - 48.0) / 64.0).to_le_bytes())
        .collect();
    gpu.hip.memcpy_htod(&part, &hp).map_err(e)?;
    gpu.hip.memset(&streams, 0, n * 8).map_err(e)?;
    gpu.hip.memset(&gates, 0, ROWS * 16).map_err(e)?;
    let start = gpu.hip.event_create().map_err(e)?;
    let stop = gpu.hip.event_create().map_err(e)?;
    let mut out = Vec::new();
    let mut tag = 0u32;
    for (label, blocks, lanes, width, pb, mode, intra, ng) in configs() {
        let launch = |gpu: &Gpu, tag: u32| -> Result<(), String> {
            let mut k = KernargBlob::new();
            k.push_ptr(peer_sink as *const c_void);
            k.push_u32(ng);
            k.push_i32(width);
            k.push_i32(pb);
            k.push_i32(mode);
            k.push_i32(intra);
            k.push_ptr(part.as_ptr());
            k.push_ptr(streams.as_ptr());
            k.push_ptr(gates.as_ptr());
            k.push_i32((ROWS * AR_HC_HIDDEN / 4) as i32);
            k.push_u32(tag);
            gpu.launch_kernel_blob(
                "plab_kernel",
                [blocks, 1, 1],
                [lanes, 1, 1],
                0,
                k.as_mut_slice(),
            )
            .map_err(|e| e.to_string())
        };
        for _ in 0..10 {
            tag += 1;
            launch(gpu, tag)?;
        }
        gpu.hip.device_synchronize().map_err(e)?;
        barrier.wait();
        let mut sk = KernargBlob::new();
        sk.push_u64(300_000);
        gpu.launch_kernel_blob("plab_spin", [1, 1, 1], [32, 1, 1], 0, sk.as_mut_slice())
            .map_err(e)?;
        gpu.hip.event_record(&start, None).map_err(e)?;
        for _ in 0..iters {
            tag += 1;
            launch(gpu, tag)?;
        }
        gpu.hip.event_record(&stop, None).map_err(e)?;
        gpu.hip.device_synchronize().map_err(e)?;
        let us =
            f64::from(gpu.hip.event_elapsed_ms(&start, &stop).map_err(e)?) * 1000.0 / iters as f64;
        out.push((label, us));
    }
    barrier.wait();
    for b in [part, streams, gates] {
        gpu.hip.free(b).map_err(e)?;
    }
    Ok(out)
}
