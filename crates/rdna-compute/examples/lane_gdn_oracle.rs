// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Byte-exactness oracle of the lane-batched DeltaNet (GDN) verify preamble +
//! recurrence of the DFlash continuous-batching trunk (exact gfx1201) against
//! today's per-lane launch sequence, for EQUAL and RAGGED (variable-row) lanes.
//!
//! 27B shapes (n_v_heads 48, n_key_heads 16, head_dim 128 -> k_dim 2048,
//! v_dim 6144, qkv_dim 10240). A case is a vector `ns` of per-lane token
//! counts: lane i owns rows `[P_i, P_i + n_i)` of ONE shared row scratch
//! (`P_i = sum_{j<i} n_j`, total `S = sum n_i`) with random pre-activation
//! qkv/alpha/beta rows, random conv rings, random Q8 state (i8 codes, small
//! positive scales) and a random f16 EF residual. Grids: the equal grid lanes
//! in {1, 3, 8, 16} x n in {1, 2, 8, 9, 16}, equal `[17]`, `[31, 31]`, `[63]`,
//! and the ragged vectors (including n = 1 lanes, a 16-lane 136-row vector and
//! the product-valid `[8,3,8,2,8,5,8,2]`), each with EF residual {present,
//! absent}; ragged vectors also at a nonzero and a near-u32-wrap frame base.
//!
//! - arm A (today): per lane, on that lane's row views:
//!   `fused_sigmoid_alpha_gate_f32_batched`, `conv1d_silu_split_f32_n`,
//!   `fused_qk_l2_norm_scale_interleave_f32_batched`,
//!   `gated_delta_net_q8_batch_seq`;
//! - arm B (lane-batched): one `fused_sigmoid_alpha_gate_f32_batched` and one
//!   `fused_qk_l2_norm_scale_interleave_f32_batched` over all rows,
//!   `Gpu::conv1d_silu_split_f32_lanes`, and one
//!   `gated_delta_net_q8_fast_lanes_varrows` over a device `GdnLaneTable`
//!   (per-lane `n_tokens` + relative `frame_off` prefix);
//! - arm C (snapshot-free): arm B's shape, but the destination bank starts as
//!   sentinel bytes and each lane's conv ring (LOGICAL ring bytes) is first
//!   copied byte-for-byte from the source bank into the destination ring, then
//!   the UNCHANGED in-place `Gpu::conv1d_silu_split_f32_lanes` runs on the
//!   destination rings (a src/dst conv twin was rejected: its codegen differs
//!   from the in-place kernel by 1-8 ulp), then
//!   `gated_delta_net_q8_fast_lanes_varrows_from` over a `GdnLaneTableFrom`
//!   table (source S q8 / scales / EF, destination bank written). The source
//!   bank holds the same initial state the other arms start from. C is only
//!   run when every n <= 16 (the sourced product contract);
//! - arms oldB / oldC (equal cases only): arms B / C through the OLD uniform
//!   replay APIs (`gated_delta_net_q8_fast_ml`, `gated_delta_net_q8_fast_ml_from`),
//!   compared against A and directly against the new B / C.
//!
//! All arms start from identical inputs and the same stochastic-rounding frame
//! base (`gdn_requant_frame_checkpoint` / `restore_...`). Every shared row
//! buffer (qkv, alpha, beta, q_raw, k_raw, v, q, k, attn_out; output buffers
//! pre-filled with a sentinel so a missed store shows) and every per-lane
//! state (conv ring, Q8 state, scales, EF residual) of the other arms is
//! compared as bytes against arm A (arm C's DESTINATION bank), and arm C's
//! SOURCE bank is asserted byte-identical to the host's initial state (never
//! written). The frame counter is read right after each arm: A (per lane
//! before/after = frame0 + P_i / frame0 + P_i + n_i, and after all lanes), B,
//! C (each must equal frame0 + S, even with EF, which bypasses the RNG).
//!
//! Negative controls (the oracle must detect them): with EF off a shifted
//! `frame_off` of one ragged row (unchanged total) changes that lane's state
//! bytes; a wrong `n_tokens` (shorter, or longer within the allocation) changes
//! the whole-buffer snapshot. The hipGraph capture/replay sub-test of the plan
//! is NOT implemented here (it needs capture/blob/exec plumbing this example
//! does not have). Prints PASS/FAIL per case; exits nonzero on any mismatch.
//!
//! `cargo run --release -p rdna-compute --features lab,deltanet --example lane_gdn_oracle`

use rdna_compute::dflash_gdn_replay::{table_bytes, GdnLayerTable, GdnLayerTableFrom};
use rdna_compute::lane_gdn::{GdnLaneTable, GdnLaneTableFrom};
use rdna_compute::norm::{gdn_requant_frame_checkpoint, restore_gdn_requant_frame_checkpoint};
use rdna_compute::tape_copy::TapeCopy;
use rdna_compute::{DType, Gpu, GpuTensor};

const NV: usize = 48;
const NK: usize = 16;
const HD: usize = 128;
const K_DIM: usize = NK * HD;
const V_DIM: usize = NV * HD;
const QKV_DIM: usize = 2 * K_DIM + V_DIM;
const EPS: f32 = 1e-6;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u32 << 24) as f32
    }
    /// Uniform in [-1, 1).
    fn sym(&mut self) -> f32 {
        self.unit() * 2.0 - 1.0
    }
}

/// Round-to-nearest-even f16 bits of a finite value with |v| in the normal
/// f16 range (callers keep magnitudes in [2^-13, 1]).
fn f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let mut exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7f_ffff;
    let mut m10 = mant >> 13;
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && m10 & 1 == 1) {
        m10 += 1;
    }
    if m10 == 0x400 {
        m10 = 0;
        exp += 1;
    }
    assert!((1..=30).contains(&exp), "{v} is not a normal f16");
    sign | ((exp as u16) << 10) | m10 as u16
}

/// Host-side initial contents shared by both arms.
struct Host {
    qkv: Vec<f32>,
    alpha: Vec<f32>,
    beta: Vec<f32>,
    conv: Vec<Vec<f32>>,
    s_q8: Vec<Vec<u8>>,
    scales: Vec<Vec<f32>>,
    ef: Vec<Vec<u8>>,
}

fn f32s(rng: &mut Rng, n: usize, amp: f32) -> Vec<f32> {
    (0..n).map(|_| rng.sym() * amp).collect()
}

fn make_host(rng: &mut Rng, lanes: usize, total: usize) -> Host {
    let qkv = f32s(rng, total * QKV_DIM, 1.5);
    let alpha = f32s(rng, total * NV, 2.0);
    let beta = f32s(rng, total * NV, 2.0);
    let conv = (0..lanes).map(|_| f32s(rng, QKV_DIM * 3, 1.0)).collect();
    let s_q8 = (0..lanes)
        .map(|_| {
            (0..NV * HD * HD)
                .map(|_| ((rng.next() % 255) as i32 - 127) as i8 as u8)
                .collect()
        })
        .collect();
    let scales = (0..lanes)
        .map(|_| (0..NV * HD).map(|_| 0.001 + rng.unit() * 0.02).collect())
        .collect();
    let ef = (0..lanes)
        .map(|_| {
            let mut b = Vec::with_capacity(NV * HD * HD * 2);
            for _ in 0..NV * HD * HD {
                let mut v = rng.sym() * 0.01;
                if v.abs() < 2f32.powi(-13) {
                    v = 0.001;
                }
                b.extend_from_slice(&f16_bits(v).to_le_bytes());
            }
            b
        })
        .collect();
    Host {
        qkv,
        alpha,
        beta,
        conv,
        s_q8,
        scales,
        ef,
    }
}

/// Per-lane recurrent state of one arm (one "bank").
struct Bank {
    conv: Vec<GpuTensor>,
    s_q8: Vec<GpuTensor>,
    scales: Vec<GpuTensor>,
    ef: Vec<Option<GpuTensor>>,
}

/// Byte pattern of an unwritten destination-bank byte (arm C).
const BANK_SENTINEL: u8 = 0xA5;

/// `sentinel == false`: the host's initial state. `true`: every byte of every
/// buffer is [`BANK_SENTINEL`] (same sizes), so a missed store shows.
fn make_bank(gpu: &mut Gpu, h: &Host, with_ef: bool, sentinel: bool) -> Bank {
    let f32_buf = |gpu: &mut Gpu, v: &[f32]| {
        if sentinel {
            let w = f32::from_bits(u32::from_ne_bytes([BANK_SENTINEL; 4]));
            gpu.upload_f32(&vec![w; v.len()], &[v.len()])
                .expect("sentinel f32")
        } else {
            gpu.upload_f32(v, &[v.len()]).expect("f32 state")
        }
    };
    let raw_buf = |gpu: &mut Gpu, v: &[u8]| {
        if sentinel {
            gpu.upload_raw(&vec![BANK_SENTINEL; v.len()], &[v.len()])
                .expect("sentinel raw")
        } else {
            gpu.upload_raw(v, &[v.len()]).expect("raw state")
        }
    };
    Bank {
        conv: h.conv.iter().map(|c| f32_buf(gpu, c)).collect(),
        s_q8: h.s_q8.iter().map(|s| raw_buf(gpu, s)).collect(),
        scales: h.scales.iter().map(|s| f32_buf(gpu, s)).collect(),
        ef: h
            .ef
            .iter()
            .map(|e| with_ef.then(|| raw_buf(gpu, e)))
            .collect(),
    }
}

/// Device buffers of one arm.
struct Arm {
    qkv: GpuTensor,
    alpha: GpuTensor,
    beta: GpuTensor,
    q_raw: GpuTensor,
    k_raw: GpuTensor,
    v: GpuTensor,
    q: GpuTensor,
    k: GpuTensor,
    attn: GpuTensor,
    conv: Vec<GpuTensor>,
    s_q8: Vec<GpuTensor>,
    scales: Vec<GpuTensor>,
    ef: Vec<Option<GpuTensor>>,
}

/// `sentinel_state`: the arm's state bank starts as [`BANK_SENTINEL`] bytes
/// (arm C's destination bank) instead of the host's initial state.
fn make_arm(gpu: &mut Gpu, h: &Host, total: usize, with_ef: bool, sentinel_state: bool) -> Arm {
    let sentinel = |gpu: &mut Gpu, w: usize| {
        gpu.upload_f32(&vec![f32::from_bits(0x7fc0_dead); total * w], &[total * w])
            .expect("alloc sentinel")
    };
    let Bank {
        conv,
        s_q8,
        scales,
        ef,
    } = make_bank(gpu, h, with_ef, sentinel_state);
    Arm {
        qkv: gpu.upload_f32(&h.qkv, &[h.qkv.len()]).expect("qkv"),
        alpha: gpu.upload_f32(&h.alpha, &[h.alpha.len()]).expect("alpha"),
        beta: gpu.upload_f32(&h.beta, &[h.beta.len()]).expect("beta"),
        q_raw: sentinel(gpu, K_DIM),
        k_raw: sentinel(gpu, K_DIM),
        v: sentinel(gpu, V_DIM),
        q: sentinel(gpu, V_DIM),
        k: sentinel(gpu, V_DIM),
        attn: sentinel(gpu, V_DIM),
        conv,
        s_q8,
        scales,
        ef,
    }
}

/// Names of the bank buffers whose bytes differ from the host's initial state
/// (`name:count`), i.e. a source bank the `_from` kernels wrote.
fn bank_vs_host(gpu: &Gpu, bank: &Bank, h: &Host) -> Vec<String> {
    gpu.hip
        .device_synchronize()
        .expect("sync before bank download");
    let mut bad = Vec::new();
    let mut chk = |name: String, t: &GpuTensor, want: &[u8]| {
        let got = gpu.download_raw_bytes(t).expect("download");
        let mism =
            got.iter().zip(want).filter(|(p, q)| p != q).count() + got.len().abs_diff(want.len());
        if mism != 0 {
            bad.push(format!("{name}:{mism}"));
        }
    };
    let f32_bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_ne_bytes()).collect::<Vec<u8>>();
    for i in 0..bank.conv.len() {
        chk(
            format!("src_conv[{i}]"),
            &bank.conv[i],
            &f32_bytes(&h.conv[i]),
        );
        chk(format!("src_s_q8[{i}]"), &bank.s_q8[i], &h.s_q8[i]);
        chk(
            format!("src_scales[{i}]"),
            &bank.scales[i],
            &f32_bytes(&h.scales[i]),
        );
        if let Some(e) = &bank.ef[i] {
            chk(format!("src_ef[{i}]"), e, &h.ef[i]);
        }
    }
    bad
}

fn snapshot(gpu: &Gpu, a: &Arm) -> Vec<(String, Vec<u8>)> {
    gpu.hip.device_synchronize().expect("sync before download");
    let mut out = Vec::new();
    let mut put = |name: String, t: &GpuTensor| {
        out.push((name, gpu.download_raw_bytes(t).expect("download")))
    };
    for (name, t) in [
        ("qkv", &a.qkv),
        ("alpha", &a.alpha),
        ("beta", &a.beta),
        ("q_raw", &a.q_raw),
        ("k_raw", &a.k_raw),
        ("v", &a.v),
        ("q", &a.q),
        ("k", &a.k),
        ("attn_out", &a.attn),
    ] {
        put(name.to_string(), t);
    }
    for i in 0..a.conv.len() {
        put(format!("conv[{i}]"), &a.conv[i]);
        put(format!("s_q8[{i}]"), &a.s_q8[i]);
        put(format!("scales[{i}]"), &a.scales[i]);
        if let Some(e) = &a.ef[i] {
            put(format!("ef[{i}]"), e);
        }
    }
    out
}

/// Shared device constants of every case.
struct Ctx {
    dt_bias: GpuTensor,
    a_log: GpuTensor,
    conv_w: GpuTensor,
}

/// Start row of each lane.
fn prefix(ns: &[usize]) -> Vec<usize> {
    let mut p = 0;
    ns.iter()
        .map(|&n| {
            let r = p;
            p += n;
            r
        })
        .collect()
}

fn free_arm(gpu: &mut Gpu, a: Arm) {
    for t in [
        a.qkv, a.alpha, a.beta, a.q_raw, a.k_raw, a.v, a.q, a.k, a.attn,
    ] {
        gpu.free_tensor(t).expect("free");
    }
    for t in a.conv.into_iter().chain(a.s_q8).chain(a.scales) {
        gpu.free_tensor(t).expect("free");
    }
    for t in a.ef.into_iter().flatten() {
        gpu.free_tensor(t).expect("free");
    }
}

fn free_bank(gpu: &mut Gpu, b: Bank) {
    for t in b.conv.into_iter().chain(b.s_q8).chain(b.scales) {
        gpu.free_tensor(t).expect("free");
    }
    for t in b.ef.into_iter().flatten() {
        gpu.free_tensor(t).expect("free");
    }
}

/// Names of the buffers whose bytes differ between two snapshots.
fn diff(x: &[(String, Vec<u8>)], y: &[(String, Vec<u8>)]) -> Vec<String> {
    assert_eq!(x.len(), y.len(), "snapshot shape");
    let mut bad = Vec::new();
    for ((name, p), (_, q)) in x.iter().zip(y) {
        let mism = p.iter().zip(q).filter(|(a, b)| a != b).count() + p.len().abs_diff(q.len());
        if mism != 0 {
            bad.push(format!("{name}:{mism}"));
        }
    }
    bad
}

/// Arm A result.
struct ARun {
    arm: Arm,
    snap: Vec<(String, Vec<u8>)>,
    /// Counter right after the last lane.
    end: u32,
    /// Lanes whose before/after counter is not frame0 + P_i / + n_i.
    ck_bad: Vec<String>,
}

/// Arm A: today's per-lane sequence on each lane's own row views.
fn run_arm_a(
    gpu: &mut Gpu,
    ctx: &Ctx,
    host: &Host,
    ns: &[usize],
    with_ef: bool,
    frame0: u32,
) -> ARun {
    let lanes = ns.len();
    let offs = prefix(ns);
    let total: usize = ns.iter().sum();
    let ratio = NV / NK;
    let q_scale = 1.0 / (HD as f32).sqrt();
    let a = make_arm(gpu, host, total, with_ef, false);
    gpu.hip.device_synchronize().expect("sync");
    restore_gdn_requant_frame_checkpoint(frame0);
    let mut ck_bad = Vec::new();
    for i in 0..lanes {
        let (r0, n) = (offs[i], ns[i]);
        let vw = |t: &GpuTensor, w: usize| t.sub_offset(r0 * w, n * w);
        let before = gdn_requant_frame_checkpoint();
        gpu.fused_sigmoid_alpha_gate_f32_batched(
            &vw(&a.beta, NV),
            &vw(&a.alpha, NV),
            &ctx.dt_bias,
            &ctx.a_log,
            NV,
            n,
        )
        .expect("A sigmoid");
        gpu.conv1d_silu_split_f32_n(
            &vw(&a.q_raw, K_DIM),
            &vw(&a.k_raw, K_DIM),
            &vw(&a.v, V_DIM),
            &vw(&a.qkv, QKV_DIM),
            &ctx.conv_w,
            &a.conv[i],
            K_DIM,
            V_DIM,
            n,
        )
        .expect("A conv");
        gpu.fused_qk_l2_norm_scale_interleave_f32_batched(
            &vw(&a.q_raw, K_DIM),
            &vw(&a.k_raw, K_DIM),
            &vw(&a.q, V_DIM),
            &vw(&a.k, V_DIM),
            NK,
            ratio,
            HD,
            q_scale,
            EPS,
            n,
        )
        .expect("A interleave");
        gpu.gated_delta_net_q8_batch_seq(
            &vw(&a.q, V_DIM),
            &vw(&a.k, V_DIM),
            &vw(&a.v, V_DIM),
            &vw(&a.alpha, NV),
            &vw(&a.beta, NV),
            &a.s_q8[i],
            &a.scales[i],
            &vw(&a.attn, V_DIM),
            n,
            NV,
            HD,
            a.ef[i].as_ref(),
        )
        .expect("A gdn");
        let after = gdn_requant_frame_checkpoint();
        let want_before = frame0.wrapping_add(r0 as u32);
        let want_after = want_before.wrapping_add(n as u32);
        if before != want_before || after != want_after {
            ck_bad.push(format!(
                "lane{i}:{before:#x}->{after:#x}(want {want_before:#x}->{want_after:#x})"
            ));
        }
    }
    let end = gdn_requant_frame_checkpoint();
    let snap = snapshot(gpu, &a);
    ARun {
        arm: a,
        snap,
        end,
        ck_bad,
    }
}

/// Lane-batched arm variants: the new variable-row ABI or the old uniform
/// replay ABI (equal ns only), each in-place or sourced (arm C shape).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Variant {
    New,
    NewFrom,
    Old,
    OldFrom,
}

impl Variant {
    fn sourced(self) -> bool {
        matches!(self, Variant::NewFrom | Variant::OldFrom)
    }
    fn name(self) -> &'static str {
        match self {
            Variant::New => "B",
            Variant::NewFrom => "C",
            Variant::Old => "oldB",
            Variant::OldFrom => "oldC",
        }
    }
}

/// Negative-control metadata overrides of the NEW table rows: `(lane, value)`.
#[derive(Default, Clone, Copy)]
struct Tweak {
    frame_off: Option<(usize, u32)>,
    n_tokens: Option<(usize, u32)>,
}

/// Result of one lane-batched variant.
struct VarRun {
    arm: Arm,
    snap: Vec<(String, Vec<u8>)>,
    /// Counter right after the recurrence launch.
    end: u32,
    /// Source-bank buffers a sourced variant wrote (must be empty).
    src_bad: Vec<String>,
}

#[allow(clippy::too_many_arguments)]
fn run_variant(
    gpu: &mut Gpu,
    ctx: &Ctx,
    host: &Host,
    ns: &[usize],
    with_ef: bool,
    v: Variant,
    frame0: u32,
    tw: Tweak,
) -> VarRun {
    let lanes = ns.len();
    let offs = prefix(ns);
    let total: usize = ns.iter().sum();
    let ratio = NV / NK;
    let q_scale = 1.0 / (HD as f32).sqrt();
    if matches!(v, Variant::Old | Variant::OldFrom) {
        assert!(
            ns.iter().all(|&n| n == ns[0]),
            "old uniform ML APIs need equal n"
        );
    }
    let arm = make_arm(gpu, host, total, with_ef, v.sourced());
    let bank_src = v.sourced().then(|| make_bank(gpu, host, with_ef, false));
    let row_size = match v {
        Variant::New => std::mem::size_of::<GdnLaneTable>(),
        Variant::NewFrom => std::mem::size_of::<GdnLaneTableFrom>(),
        Variant::Old => std::mem::size_of::<GdnLayerTable>(),
        Variant::OldFrom => std::mem::size_of::<GdnLayerTableFrom>(),
    };
    let table = gpu
        .zeros(&[(lanes * row_size).div_ceil(4)], DType::F32)
        .expect("table");
    gpu.hip.device_synchronize().expect("sync");
    restore_gdn_requant_frame_checkpoint(frame0);

    gpu.fused_sigmoid_alpha_gate_f32_batched(
        &arm.beta,
        &arm.alpha,
        &ctx.dt_bias,
        &ctx.a_log,
        NV,
        total,
    )
    .expect("sigmoid");
    if let Some(src) = &bank_src {
        // Conv ring: byte copy src bank -> destination ring, then the
        // unchanged in-place lane conv on the destination (the `_from` conv
        // twin's codegen differed by a few ulp). LOGICAL ring bytes (the pool
        // may round the allocations up).
        let ring_copies: Vec<TapeCopy> = (0..lanes)
            .map(|i| {
                TapeCopy::new(
                    &arm.conv[i].buf,
                    0,
                    &src.conv[i].buf,
                    0,
                    host.conv[i].len() * 4,
                )
            })
            .collect();
        gpu.tape_copy_batched(&ring_copies).expect("ring copy");
    }
    let lane_args: Vec<(u64, usize, usize)> = (0..lanes)
        .map(|i| (arm.conv[i].buf.as_ptr() as u64, offs[i], ns[i]))
        .collect();
    gpu.conv1d_silu_split_f32_lanes(
        &arm.q_raw,
        &arm.k_raw,
        &arm.v,
        &arm.qkv,
        &ctx.conv_w,
        &lane_args,
        K_DIM,
        V_DIM,
    )
    .expect("conv");
    gpu.fused_qk_l2_norm_scale_interleave_f32_batched(
        &arm.q_raw, &arm.k_raw, &arm.q, &arm.k, NK, ratio, HD, q_scale, EPS, total,
    )
    .expect("interleave");

    let ptr = |t: &GpuTensor| t.buf.as_ptr() as u64;
    let base_rows: Vec<GdnLayerTable> = (0..lanes)
        .map(|i| {
            let vw = |t: &GpuTensor, w: usize| t.sub_offset(offs[i] * w, ns[i] * w);
            GdnLayerTable {
                q: ptr(&vw(&arm.q, V_DIM)),
                k: ptr(&vw(&arm.k, V_DIM)),
                v: ptr(&vw(&arm.v, V_DIM)),
                gate: ptr(&vw(&arm.alpha, NV)),
                beta: ptr(&vw(&arm.beta, NV)),
                s_q8: ptr(&arm.s_q8[i]),
                s_scales: ptr(&arm.scales[i]),
                output: ptr(&vw(&arm.attn, V_DIM)),
                ef: arm.ef[i].as_ref().map_or(0, ptr),
            }
        })
        .collect();
    let from_rows: Vec<GdnLayerTableFrom> = match &bank_src {
        Some(src) => (0..lanes)
            .map(|i| GdnLayerTableFrom {
                base: base_rows[i],
                s_q8_src: ptr(&src.s_q8[i]),
                s_scales_src: ptr(&src.scales[i]),
                ef_src: src.ef[i].as_ref().map_or(0, ptr),
            })
            .collect(),
        None => Vec::new(),
    };
    let n_tok = |i: usize| match tw.n_tokens {
        Some((l, n)) if l == i => n,
        _ => ns[i] as u32,
    };
    let f_off = |i: usize| match tw.frame_off {
        Some((l, f)) if l == i => f,
        _ => offs[i] as u32,
    };
    match v {
        Variant::Old => {
            gpu.upload_gdn_lane_table(&table, table_bytes(&base_rows))
                .expect("table upload");
            gpu.gated_delta_net_q8_fast_ml(table.buf.as_ptr(), lanes, ns[0], NV, HD)
                .expect("old gdn");
        }
        Variant::OldFrom => {
            gpu.upload_gdn_lane_table(&table, table_bytes(&from_rows))
                .expect("table upload");
            gpu.gated_delta_net_q8_fast_ml_from(table.buf.as_ptr(), lanes, ns[0], NV, HD)
                .expect("old gdn_from");
        }
        Variant::New => {
            let rows: Vec<GdnLaneTable> = (0..lanes)
                .map(|i| GdnLaneTable {
                    base: base_rows[i],
                    n_tokens: n_tok(i),
                    frame_off: f_off(i),
                })
                .collect();
            gpu.upload_gdn_lane_table(&table, table_bytes(&rows))
                .expect("table upload");
            gpu.gated_delta_net_q8_fast_lanes_varrows(table.buf.as_ptr(), lanes, total, NV, HD)
                .expect("varrows gdn");
        }
        Variant::NewFrom => {
            let rows: Vec<GdnLaneTableFrom> = (0..lanes)
                .map(|i| GdnLaneTableFrom {
                    base: from_rows[i],
                    n_tokens: n_tok(i),
                    frame_off: f_off(i),
                })
                .collect();
            gpu.upload_gdn_lane_table(&table, table_bytes(&rows))
                .expect("table upload");
            gpu.gated_delta_net_q8_fast_lanes_varrows_from(
                table.buf.as_ptr(),
                lanes,
                total,
                NV,
                HD,
            )
            .expect("varrows gdn_from");
        }
    }
    let end = gdn_requant_frame_checkpoint();
    let snap = snapshot(gpu, &arm);
    let src_bad = bank_src
        .as_ref()
        .map(|b| bank_vs_host(gpu, b, host))
        .unwrap_or_default();
    gpu.free_tensor(table).expect("free");
    if let Some(b) = bank_src {
        free_bank(gpu, b);
    }
    VarRun {
        arm,
        snap,
        end,
        src_bad,
    }
}

/// One case: arm A vs new B (and C when every n <= 16), plus the old uniform
/// ML B / C when the lanes are equal. `base`: restore the frame counter to it
/// first (else continue from the current value).
fn run_case(
    gpu: &mut Gpu,
    ctx: &Ctx,
    rng: &mut Rng,
    ns: &[usize],
    with_ef: bool,
    base: Option<u32>,
) -> bool {
    let lanes = ns.len();
    let total: usize = ns.iter().sum();
    let equal = ns.iter().all(|&n| n == ns[0]);
    let sourced_ok = ns.iter().all(|&n| n <= 16);
    if let Some(b) = base {
        restore_gdn_requant_frame_checkpoint(b);
    }
    let frame0 = gdn_requant_frame_checkpoint();
    let want_end = frame0.wrapping_add(total as u32);
    let host = make_host(rng, lanes, total);

    let a = run_arm_a(gpu, ctx, &host, ns, with_ef, frame0);
    let mut ok = true;
    let mut line = format!(
        "ns={ns:?} rows={total} ef={with_ef:<5} frame0={frame0:#010x} | A end_ok={} lane_ck=[{}]",
        a.end == want_end,
        a.ck_bad.join(", ")
    );
    ok &= a.end == want_end && a.ck_bad.is_empty();

    let mut variants = vec![Variant::New];
    if sourced_ok {
        variants.push(Variant::NewFrom);
    }
    if equal {
        variants.push(Variant::Old);
        if sourced_ok {
            variants.push(Variant::OldFrom);
        }
    }
    let mut runs: Vec<(Variant, VarRun)> = Vec::new();
    for &v in &variants {
        let r = run_variant(gpu, ctx, &host, ns, with_ef, v, frame0, Tweak::default());
        let bad = diff(&a.snap, &r.snap);
        let end_ok = r.end == want_end;
        ok &= bad.is_empty() && end_ok && r.src_bad.is_empty();
        line.push_str(&format!(
            " | {} end_ok={end_ok} mismatching_buffers=[{}]",
            v.name(),
            bad.join(", ")
        ));
        if v.sourced() {
            line.push_str(&format!(" source_bank_changed=[{}]", r.src_bad.join(", ")));
        }
        runs.push((v, r));
    }
    // New vs old uniform ML APIs, directly.
    for (new_v, old_v) in [
        (Variant::New, Variant::Old),
        (Variant::NewFrom, Variant::OldFrom),
    ] {
        let find = |want: Variant| runs.iter().find(|(v, _)| *v == want).map(|(_, r)| r);
        if let (Some(n), Some(o)) = (find(new_v), find(old_v)) {
            let bad = diff(&o.snap, &n.snap);
            ok &= bad.is_empty();
            line.push_str(&format!(
                " | {}-vs-{} mismatching_buffers=[{}]",
                new_v.name(),
                old_v.name(),
                bad.join(", ")
            ));
        }
    }
    println!("{} {line}", if ok { "PASS" } else { "FAIL" });

    free_arm(gpu, a.arm);
    for (_, r) in runs {
        free_arm(gpu, r.arm);
    }
    ok
}

/// Negative controls: corrupted table metadata must be detected. Returns the
/// number of failed controls.
fn negative_controls(gpu: &mut Gpu, ctx: &Ctx, rng: &mut Rng) -> usize {
    let ns = [8usize, 3, 8, 1, 8, 5, 8, 2];
    let offs = prefix(&ns);
    let total: usize = ns.iter().sum();
    restore_gdn_requant_frame_checkpoint(0x1357_9BDF);
    let frame0 = gdn_requant_frame_checkpoint();
    let want_end = frame0.wrapping_add(total as u32);
    let host = make_host(rng, ns.len(), total);
    let mut failures = 0usize;
    for with_ef in [false, true] {
        let a = run_arm_a(gpu, ctx, &host, &ns, with_ef, frame0);
        for v in [Variant::New, Variant::NewFrom] {
            let mut report = |gpu: &mut Gpu,
                              name: String,
                              tw: Tweak,
                              expect: &dyn Fn(&[String]) -> bool,
                              must_detect: bool| {
                let r = run_variant(gpu, ctx, &host, &ns, with_ef, v, frame0, tw);
                let bad = diff(&a.snap, &r.snap);
                // The total reservation is unchanged by the corruption.
                let end_ok = r.end == want_end;
                let detected = expect(&bad);
                let pass = end_ok && (detected == must_detect);
                if !pass {
                    failures += 1;
                }
                println!(
                    "{} control {} ef={with_ef:<5} {name}: detected={detected} (must_detect={must_detect}) end_ok={end_ok} mismatching_buffers=[{}]",
                    if pass { "PASS" } else { "FAIL" },
                    v.name(),
                    bad.join(", ")
                );
                free_arm(gpu, r.arm);
            };
            // Baseline: untouched metadata must match A.
            report(
                gpu,
                "clean metadata".into(),
                Tweak::default(),
                &|bad| !bad.is_empty(),
                false,
            );
            // Shifted frame prefix: only the EF-off stochastic seed sees it.
            for lane in [1usize, 4, 7] {
                let tw = Tweak {
                    frame_off: Some((lane, offs[lane] as u32 + 1)),
                    ..Tweak::default()
                };
                let state = format!("s_q8[{lane}]:");
                let scales = format!("scales[{lane}]:");
                if with_ef {
                    // EF bypasses the RNG: the shifted prefix must change nothing.
                    report(
                        gpu,
                        format!("frame_off[{lane}]+1 (EF bypasses RNG)"),
                        tw,
                        &|bad| !bad.is_empty(),
                        false,
                    );
                } else {
                    report(
                        gpu,
                        format!("frame_off[{lane}]+1"),
                        tw,
                        &|bad| {
                            bad.iter()
                                .any(|m| m.starts_with(&state) || m.starts_with(&scales))
                        },
                        true,
                    );
                }
            }
            // Wrong token counts inside the allocation (shorter / longer).
            for (lane, n) in [(0usize, 7u32), (1, 4)] {
                let tw = Tweak {
                    n_tokens: Some((lane, n)),
                    ..Tweak::default()
                };
                report(
                    gpu,
                    format!("n_tokens[{lane}]={n} (true {})", ns[lane]),
                    tw,
                    &|bad| !bad.is_empty(),
                    true,
                );
            }
        }
        free_arm(gpu, a.arm);
    }
    failures
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    if gpu.arch != "gfx1201" {
        eprintln!("lane_gdn_oracle: needs exact gfx1201, got {}", gpu.arch);
        std::process::exit(2);
    }
    if !gpu.lane_gdn_batch_admitted(NV, NK, HD, HD) {
        eprintln!(
            "lane_gdn_oracle: lane_gdn_batch_admitted is false (HIPFIRE_DN_REQUANT_PER_TOKEN set?)"
        );
        std::process::exit(2);
    }
    println!("arch={} lane_gdn_batch_admitted=true", gpu.arch);

    let mut rng = Rng(0x6D1E_47A3);
    let dt_bias: Vec<f32> = (0..NV).map(|_| rng.sym() * 0.5).collect();
    let a_log: Vec<f32> = (0..NV).map(|_| rng.sym()).collect();
    let conv_w: Vec<f32> = (0..QKV_DIM * 4).map(|_| rng.sym() * 0.5).collect();
    let ctx = Ctx {
        dt_bias: gpu.upload_f32(&dt_bias, &[NV]).expect("dt_bias"),
        a_log: gpu.upload_f32(&a_log, &[NV]).expect("a_log"),
        conv_w: gpu.upload_f32(&conv_w, &[conv_w.len()]).expect("conv_w"),
    };

    let mut failures = 0usize;
    let mut cases = 0usize;

    // Equal grid, then the equal large cases (A vs new B vs old ML only).
    let mut equal: Vec<Vec<usize>> = Vec::new();
    for &lanes in &[1usize, 3, 8, 16] {
        for &n in &[1usize, 2, 8, 9, 16] {
            equal.push(vec![n; lanes]);
        }
    }
    equal.extend([vec![17], vec![31, 31], vec![63]]);
    for ns in &equal {
        for &with_ef in &[true, false] {
            cases += 1;
            if !run_case(&mut gpu, &ctx, &mut rng, ns, with_ef, None) {
                failures += 1;
            }
        }
    }

    // Ragged cases: both EF states, a nonzero and a near-u32-wrap frame base.
    let ragged: [&[usize]; 7] = [
        &[8, 3, 8, 1, 8, 5, 8, 2],
        &[2, 8, 5, 8, 1, 8, 3, 8],
        &[1, 16],
        &[16, 1],
        &[2, 3, 16],
        &[16, 2, 15, 3, 14, 4, 13, 5, 12, 6, 11, 7, 10, 8, 9, 1],
        &[8, 3, 8, 2, 8, 5, 8, 2],
    ];
    for ns in ragged {
        for &base in &[0x00C0_FFEEu32, u32::MAX - 20] {
            for &with_ef in &[false, true] {
                cases += 1;
                if !run_case(&mut gpu, &ctx, &mut rng, ns, with_ef, Some(base)) {
                    failures += 1;
                }
            }
        }
    }

    let control_failures = negative_controls(&mut gpu, &ctx, &mut rng);
    failures += control_failures;

    println!("hipGraph capture/replay sub-test: not implemented (needs capture/blob/exec plumbing this example lacks)");
    println!("cases={cases} control_failures={control_failures} failures={failures}");
    if failures != 0 {
        std::process::exit(1);
    }
}
