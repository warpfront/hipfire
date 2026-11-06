// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1201: the dense-GDN prefill chain at the TP2 trunk shard's rank-local geometry (8 key /
//! 24 value heads of 128, 5120 convolution channels) must produce, for each rank, exactly the
//! bytes of the Flash-Next chain (16 / 48 heads, 10240 channels) restricted to that rank's
//! heads. Every head of the chain is independent (per-channel convolution, per-head Q/K norm,
//! gate parameters, KKT solve, scan and state requantization; the Q8 exit rounding is
//! nearest-even, no RNG seeded by the head index), and the shard keeps whole key heads with
//! their three value heads (value head `v` reads key head `v / 3`), so rank `r`'s local head
//! `h` is global head `(8 r | 24 r) + h` and the local q | k | v columns are
//! `q[1024 r..+1024] | k[1024 r..+1024] | v[3072 r..+3072]` of the global ones.
//!
//! For ranks 0 and 1, rows {64, 65, 130, 777, 1024, 4160}, an F32 and a BF16 convolution input,
//! on random inputs with injected ties, subnormals, +-0 and extreme magnitudes and all-zero
//! head rows, every byte the chain stores is compared against the global chain's slice: the
//! packed F16 q / k / v planes, g, beta, the conv history ring, the Q8 state codes and scales,
//! and the scan's BF16 output plane; then the gated RMSNorm + rotation
//! (`fn_gdn_dense::gate`, F32 output and F32 FWHT rotation) and the fused gate + A8 producer
//! (`fn_gdn_dense::gate_i8` -> `fn_gdn_dense_gate_i8_p12` vs the global
//! `fn_gdn_dense_gate_i8`) on the scan's output. `block_i8_128` block `b` of rank `r` is block
//! `24 r + b` of the global array (a 256-wide rotation group is a head pair; rank cuts fall on
//! pair boundaries).
//!
//! `#[ignore]`d: needs an exact gfx1201 with a working HIP toolchain.
//!
//!   cargo test -p rdna-compute --release --features deltanet \
//!       --test fn_gdn_shard_oracle -- --ignored --test-threads=1 --nocapture

#![cfg(feature = "deltanet")]

use rdna_compute::fn_gdn_dense;
use rdna_compute::tensor_ops::{
    GatedDeltaConvBatched, GatedDeltaGateBatched, GatedDeltaParamsBatched, GatedDeltaStepBatched,
    GdnStateFormat,
};
use rdna_compute::{DType, Gpu, GpuTensor};

const DIM: usize = 128;
/// Flash-Next global geometry.
const G_KH: usize = 16;
const G_VH: usize = 48;
const G_QK: usize = G_KH * DIM;
const G_VW: usize = G_VH * DIM;
const G_CH: usize = 2 * G_QK + G_VW;
/// Rank-local geometry of the TP2 shard (`parts` = 2).
const L_KH: usize = G_KH / 2;
const L_VH: usize = G_VH / 2;
const L_QK: usize = L_KH * DIM;
const L_VW: usize = L_VH * DIM;
const L_CH: usize = 2 * L_QK + L_VW;
const BLOCK_BYTES: usize = 136;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// Host twin of `hip_bfloat16`'s float -> bf16 (`float_to_bfloat16`).
fn bf16_hip(f: f32) -> u16 {
    let mut u = f.to_bits();
    if (!u) & 0x7f80_0000 != 0 {
        u = u.wrapping_add(0x7fff + ((u >> 16) & 1));
    } else if u & 0xffff != 0 {
        u |= 0x1_0000;
    }
    (u >> 16) as u16
}

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn bytes_u16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

const FINITE_SPECIAL: [u32; 12] = [
    0x0000_0000,
    0x8000_0000,
    0x0000_0001,
    0x8000_0001,
    0x007f_ffff,
    0x807f_ffff,
    0x3f80_8000,
    0x3f80_7fff,
    0x3f80_8001,
    0xbf81_8000,
    0x7f7f_ffff,
    0xff7f_ffff,
];

/// Global column of rank `rank`'s local convolution channel `c` (q | k | v order).
fn global_channel(rank: usize, c: usize) -> usize {
    if c < L_QK {
        rank * L_QK + c
    } else if c < 2 * L_QK {
        G_QK + rank * L_QK + (c - L_QK)
    } else {
        2 * G_QK + rank * L_VW + (c - 2 * L_QK)
    }
}

/// The global (Flash-Next) inputs of one chain launch.
struct Case {
    rows: usize,
    input: Vec<f32>,
    kernel: Vec<u16>,
    history: Vec<f32>,
    a: Vec<f32>,
    b: Vec<f32>,
    a_log: Vec<u16>,
    dt_bias: Vec<u16>,
    /// Q8 slot: codes `[head][row][key]` then F32 scales `[head][row]`.
    state: Vec<u8>,
    z: Vec<f32>,
    norm: Vec<u16>,
    start_cursor: usize,
}

fn gen_case(rows: usize, seed: u64) -> Case {
    let mut r = Rng(seed | 1);
    let mut input = vec![0f32; rows * G_CH];
    for (row, line) in input.chunks_mut(G_CH).enumerate() {
        for (c, x) in line.iter_mut().enumerate() {
            let scale = match c / DIM {
                0..=31 => 2.0 + (c / DIM) as f32 * 0.25,
                _ => 1.0 + ((c / DIM) % 5) as f32,
            };
            *x = r.unit() * scale;
            if r.below(150) == 0 {
                *x = f32::from_bits(FINITE_SPECIAL[r.below(FINITE_SPECIAL.len() as u64) as usize]);
            }
        }
        // All-zero head rows keep the zero-norm path alive (a q|k head of either rank).
        if row % 41 == 3 {
            let head = (row / 41) % 32;
            line[head * DIM..head * DIM + DIM].fill(0.0);
        }
    }
    let kernel: Vec<u16> = (0..G_CH * 4).map(|_| bf16_hip(r.unit() * 0.6)).collect();
    let history: Vec<f32> = (0..3 * G_CH)
        .map(|_| bf16_to_f32(bf16_hip(r.unit() * 2.0)))
        .collect();
    let a: Vec<f32> = (0..rows * G_VH).map(|_| r.unit() * 8.0).collect();
    let b: Vec<f32> = (0..rows * G_VH).map(|_| r.unit() * 6.0).collect();
    let a_log: Vec<u16> = (0..G_VH).map(|_| bf16_hip(r.unit() * 1.5)).collect();
    let dt_bias: Vec<u16> = (0..G_VH).map(|_| bf16_hip(r.unit() * 2.0)).collect();
    let state_bytes = GdnStateFormat::Q8.state_bytes(G_VH, DIM, DIM);
    let mut state = vec![0u8; state_bytes];
    for byte in state[..G_VH * DIM * DIM].iter_mut() {
        *byte = (r.next() >> 56) as u8;
    }
    for s in 0..G_VH * DIM {
        let off = G_VH * DIM * DIM + s * 4;
        state[off..off + 4].copy_from_slice(&(0.002 + r.unit().abs() * 0.01f32).to_le_bytes());
    }
    let z: Vec<f32> = (0..rows * G_VW).map(|_| r.unit() * 6.0).collect();
    let norm: Vec<u16> = (0..DIM)
        .map(|_| bf16_hip(0.25 + 1.5 * (r.unit() * 0.5 + 0.5)))
        .collect();
    Case {
        rows,
        input,
        kernel,
        history,
        a,
        b,
        a_log,
        dt_bias,
        state,
        z,
        norm,
        start_cursor: (seed % 3) as usize,
    }
}

/// Rank `rank`'s local case: the same values, cut by the shard's head mapping.
fn local_case(g: &Case, rank: usize) -> Case {
    let n = g.rows;
    let mut input = vec![0f32; n * L_CH];
    for row in 0..n {
        for c in 0..L_CH {
            input[row * L_CH + c] = g.input[row * G_CH + global_channel(rank, c)];
        }
    }
    let mut kernel = vec![0u16; L_CH * 4];
    for c in 0..L_CH {
        let gc = global_channel(rank, c);
        kernel[c * 4..c * 4 + 4].copy_from_slice(&g.kernel[gc * 4..gc * 4 + 4]);
    }
    let mut history = vec![0f32; 3 * L_CH];
    for slot in 0..3 {
        for c in 0..L_CH {
            history[slot * L_CH + c] = g.history[slot * G_CH + global_channel(rank, c)];
        }
    }
    let head0 = rank * L_VH;
    let per_row = |v: &[f32]| -> Vec<f32> {
        (0..n)
            .flat_map(|row| v[row * G_VH + head0..row * G_VH + head0 + L_VH].to_vec())
            .collect()
    };
    let mut state = Vec::new();
    state.extend_from_slice(&g.state[head0 * DIM * DIM..(head0 + L_VH) * DIM * DIM]);
    let scales = G_VH * DIM * DIM;
    state.extend_from_slice(&g.state[scales + head0 * DIM * 4..scales + (head0 + L_VH) * DIM * 4]);
    let z: Vec<f32> = (0..n)
        .flat_map(|row| g.z[row * G_VW + rank * L_VW..row * G_VW + (rank + 1) * L_VW].to_vec())
        .collect();
    Case {
        rows: n,
        input,
        kernel,
        history,
        a: per_row(&g.a),
        b: per_row(&g.b),
        a_log: g.a_log[head0..head0 + L_VH].to_vec(),
        dt_bias: g.dt_bias[head0..head0 + L_VH].to_vec(),
        state,
        z,
        norm: g.norm.clone(),
        start_cursor: g.start_cursor,
    }
}

fn typed(t: GpuTensor, dtype: DType, shape: Vec<usize>) -> GpuTensor {
    GpuTensor { dtype, shape, ..t }
}

/// One chain launch's operands at head geometry `(kh, vh)`.
struct Operands {
    kh: usize,
    vh: usize,
    ch: usize,
    input: GpuTensor,
    kernel: GpuTensor,
    history: GpuTensor,
    a: GpuTensor,
    b: GpuTensor,
    a_log: GpuTensor,
    dt_bias: GpuTensor,
    gate: GpuTensor,
    beta: GpuTensor,
    proj: GpuTensor,
    state: GpuTensor,
    out: GpuTensor,
    z: GpuTensor,
    norm: GpuTensor,
    gated: GpuTensor,
    rotated: GpuTensor,
}

fn operands(gpu: &mut Gpu, c: &Case, kh: usize, vh: usize, bf16_input: bool) -> Operands {
    let n = c.rows;
    let ch = 2 * kh * DIM + vh * DIM;
    let vw = vh * DIM;
    let input = if bf16_input {
        let bits: Vec<u16> = c.input.iter().map(|&x| bf16_hip(x)).collect();
        let t = gpu
            .upload_raw(&bytes_u16(&bits), &[n * ch * 2])
            .expect("input bf16");
        typed(t, DType::BF16, vec![n * ch])
    } else {
        gpu.upload_f32(&c.input, &[n * ch]).expect("input f32")
    };
    let kernel = {
        let t = gpu
            .upload_raw(&bytes_u16(&c.kernel), &[ch * 4 * 2])
            .expect("kernel");
        typed(t, DType::BF16, vec![ch * 4])
    };
    let history = gpu.upload_f32(&c.history, &[3 * ch]).expect("history");
    let a = gpu.upload_f32(&c.a, &[n * vh]).expect("a");
    let b = gpu.upload_f32(&c.b, &[n * vh]).expect("b");
    let a_log = typed(
        gpu.upload_raw(&bytes_u16(&c.a_log), &[vh * 2])
            .expect("a_log"),
        DType::BF16,
        vec![vh],
    );
    let dt_bias = typed(
        gpu.upload_raw(&bytes_u16(&c.dt_bias), &[vh * 2])
            .expect("dt"),
        DType::BF16,
        vec![vh],
    );
    let state_bytes = GdnStateFormat::Q8.state_bytes(vh, DIM, DIM);
    assert_eq!(c.state.len(), state_bytes);
    let state = typed(
        gpu.upload_raw(&c.state, &[state_bytes]).expect("state"),
        DType::Raw,
        vec![state_bytes],
    );
    let norm = typed(
        gpu.upload_raw(&bytes_u16(&c.norm), &[DIM * 2])
            .expect("norm"),
        DType::BF16,
        vec![DIM],
    );
    Operands {
        kh,
        vh,
        ch,
        input,
        kernel,
        history,
        a,
        b,
        a_log,
        dt_bias,
        gate: gpu.zeros(&[n * vh], DType::F32).expect("gate"),
        beta: gpu.zeros(&[n * vh], DType::F32).expect("beta"),
        proj: gpu.zeros(&[n * ch], DType::F32).expect("proj"),
        state,
        out: gpu.zeros(&[n * vw], DType::F32).expect("out"),
        z: gpu.upload_f32(&c.z, &[n * vw]).expect("z"),
        norm,
        gated: gpu.zeros(&[n * vw], DType::F32).expect("gated"),
        rotated: gpu.zeros(&[n * vw], DType::F32).expect("rotated"),
    }
}

fn run_chain(gpu: &mut Gpu, o: &Operands, c: &Case) {
    let n = c.rows;
    let conv = GatedDeltaConvBatched {
        input: &o.input,
        kernel: &o.kernel,
        history: &o.history,
        output: &o.proj,
        next_history: &o.history,
        rows: n,
        channels: o.ch,
        history_rows: 3,
        kernel_size: 4,
        start_cursor: c.start_cursor,
    };
    let params = GatedDeltaParamsBatched {
        a: &o.a,
        b: &o.b,
        a_log: &o.a_log,
        dt_bias: &o.dt_bias,
        gate: &o.gate,
        beta: &o.beta,
        rows: n,
        heads: o.vh,
    };
    let step = GatedDeltaStepBatched {
        projection: &o.proj,
        gate: &o.gate,
        beta: &o.beta,
        state: &o.state,
        format: GdnStateFormat::Q8,
        residual: None,
        output: &o.out,
        row_states: None,
        rows: n,
        qkv_width: o.ch,
        key_heads: o.kh,
        value_heads: o.vh,
        key_dim: DIM,
        value_dim: DIM,
        position: 0,
    };
    fn_gdn_dense::run(gpu, &conv, &params, &step).expect("fn_gdn_dense::run");
}

/// Everything one chain stores, as bytes.
struct Outputs {
    /// Scan output plane (BF16 bits), `[rows][vh * 128]`.
    out: Vec<u8>,
    state: Vec<u8>,
    /// q | k | v planes (F16 bits) of the projection scratch, `[n64][qk] [n64][qk] [n64][vw]`.
    planes: Vec<u8>,
    gate: Vec<u8>,
    beta: Vec<u8>,
    history: Vec<u8>,
    /// Gated RMSNorm output and its FWHT rotation (F32).
    gated: Vec<u8>,
    rotated: Vec<u8>,
    /// `fn_gdn_dense::gate_i8` blocks, `[vh * 128 / 128][rows]` of 136 B.
    blocks: Vec<u8>,
}

fn download(gpu: &mut Gpu, o: &Operands, rows: usize) -> Outputs {
    let vw = o.vh * DIM;
    let qk = o.kh * DIM;
    // The gate over the scan's BF16 plane: F32 output + rotation, then the fused A8 producer.
    let gate_args = GatedDeltaGateBatched {
        recurrent_output: &o.out,
        z: &o.z,
        norm: &o.norm,
        output: &o.gated,
        rows,
        value_heads: o.vh,
        value_dim: DIM,
    };
    fn_gdn_dense::gate(gpu, &gate_args, Some(&o.rotated)).expect("gate");
    gpu.hip.device_synchronize().expect("sync");
    let gated = gpu.download_raw_bytes(&o.gated).expect("gated");
    let rotated = gpu.download_raw_bytes(&o.rotated).expect("rotated");
    let block_bytes = (vw / 128) * rows * BLOCK_BYTES;
    let ptr = fn_gdn_dense::gate_i8(gpu, &gate_args).expect("gate_i8");
    gpu.hip.device_synchronize().expect("sync");
    let mut blocks = vec![0u8; block_bytes];
    // SAFETY: `ptr` is the live A8 scratch of `gate_i8` with at least `block_bytes` bytes.
    let buf = unsafe { hip_bridge::DeviceBuffer::from_raw(ptr, block_bytes) };
    gpu.hip.memcpy_dtoh(&mut blocks, &buf).expect("blocks");
    let mut out = gpu.download_raw_bytes(&o.out).expect("out");
    out.truncate(rows * vw * 2);
    let n64 = rows.div_ceil(64) * 64;
    let mut planes = gpu.download_raw_bytes(&o.proj).expect("proj");
    planes.truncate((2 * n64 * qk + n64 * vw) * 2);
    Outputs {
        out,
        state: gpu.download_raw_bytes(&o.state).expect("state"),
        planes,
        gate: gpu.download_raw_bytes(&o.gate).expect("gate"),
        beta: gpu.download_raw_bytes(&o.beta).expect("beta"),
        history: gpu.download_raw_bytes(&o.history).expect("history"),
        gated,
        rotated,
        blocks,
    }
}

fn free(gpu: &mut Gpu, o: Operands) {
    for t in [
        o.input, o.kernel, o.history, o.a, o.b, o.a_log, o.dt_bias, o.gate, o.beta, o.proj,
        o.state, o.out, o.z, o.norm, o.gated, o.rotated,
    ] {
        gpu.free_tensor(t).expect("free");
    }
}

fn require_gfx1201() -> Option<Gpu> {
    let gpu = Gpu::init().expect("gpu init");
    if gpu.arch != "gfx1201" {
        eprintln!("skip: needs gfx1201, found {}", gpu.arch);
        return None;
    }
    Some(gpu)
}

fn set_chain_fuse(gpu: &mut Gpu, on: bool) {
    std::sync::Arc::make_mut(&mut gpu.flags).qwen4_gdn_chain_fuse = on;
    assert_eq!(gpu.flags.qwen4_gdn_chain_fuse_enabled(), on);
}

/// Compare `l` (rank `rank`'s local chain) against the rank-`rank` slice of the global chain's
/// outputs `g`; returns the bytes compared.
fn compare(what: &str, rank: usize, rows: usize, g: &Outputs, l: &Outputs) -> usize {
    let n64 = rows.div_ceil(64) * 64;
    let mut total = 0;
    let mut check = |name: &str, expect: Vec<u8>, got: &[u8]| {
        assert_eq!(
            expect.len(),
            got.len(),
            "{what} rank {rank} rows {rows} {name}: length"
        );
        let bad = expect.iter().zip(got).filter(|(p, q)| p != q).count();
        let first = expect.iter().zip(got).position(|(p, q)| p != q);
        assert_eq!(
            bad,
            0,
            "{what} rank {rank} rows {rows} {name}: {bad} of {} bytes differ (first at {first:?})",
            expect.len()
        );
        total += expect.len();
    };
    // `cols` bytes of rank `rank`'s slice of each `stride`-byte global row (`rows` rows).
    let cut = |src: &[u8], rows: usize, stride: usize, off: usize, cols: usize| -> Vec<u8> {
        (0..rows)
            .flat_map(|r| src[r * stride + off..r * stride + off + cols].to_vec())
            .collect()
    };
    check(
        "scan out",
        cut(&g.out, rows, G_VW * 2, rank * L_VW * 2, L_VW * 2),
        &l.out,
    );
    check(
        "gated out",
        cut(&g.gated, rows, G_VW * 4, rank * L_VW * 4, L_VW * 4),
        &l.gated,
    );
    check(
        "rotated out",
        cut(&g.rotated, rows, G_VW * 4, rank * L_VW * 4, L_VW * 4),
        &l.rotated,
    );
    let (gq, gv) = (n64 * G_QK * 2, n64 * G_VW * 2);
    let mut planes = Vec::new();
    planes.extend(cut(
        &g.planes[..gq],
        n64,
        G_QK * 2,
        rank * L_QK * 2,
        L_QK * 2,
    ));
    planes.extend(cut(
        &g.planes[gq..2 * gq],
        n64,
        G_QK * 2,
        rank * L_QK * 2,
        L_QK * 2,
    ));
    planes.extend(cut(
        &g.planes[2 * gq..2 * gq + gv],
        n64,
        G_VW * 2,
        rank * L_VW * 2,
        L_VW * 2,
    ));
    check("q/k/v planes", planes, &l.planes);
    check(
        "g",
        cut(&g.gate, rows, G_VH * 4, rank * L_VH * 4, L_VH * 4),
        &l.gate,
    );
    check(
        "beta",
        cut(&g.beta, rows, G_VH * 4, rank * L_VH * 4, L_VH * 4),
        &l.beta,
    );
    let mut history = Vec::new();
    for slot in 0..3 {
        for c in 0..L_CH {
            let gc = global_channel(rank, c);
            history
                .extend_from_slice(&g.history[(slot * G_CH + gc) * 4..(slot * G_CH + gc) * 4 + 4]);
        }
    }
    check("conv history", history, &l.history);
    let mut state = g.state[rank * L_VH * DIM * DIM..(rank + 1) * L_VH * DIM * DIM].to_vec();
    let scales = G_VH * DIM * DIM;
    state.extend_from_slice(
        &g.state[scales + rank * L_VH * DIM * 4..scales + (rank + 1) * L_VH * DIM * 4],
    );
    check("Q8 state", state, &l.state);
    // Block `b` (128 columns) of rank `rank` is global block `24 rank + b`; each block holds
    // `rows` consecutive 136-byte entries.
    let block_stride = rows * BLOCK_BYTES;
    let first = rank * (L_VW / 128);
    check(
        "gate_i8 blocks",
        g.blocks[first * block_stride..(first + L_VW / 128) * block_stride].to_vec(),
        &l.blocks,
    );
    total
}

#[test]
#[ignore = "needs an exact gfx1201"]
fn shard_chain_is_bytewise_the_flash_next_chain_restricted_to_the_rank_heads() {
    let Some(mut gpu) = require_gfx1201() else {
        return;
    };
    set_chain_fuse(&mut gpu, true);
    for (n, rows) in [64usize, 65, 130, 777, 1024, 4160].into_iter().enumerate() {
        let case = gen_case(rows, 0x5a4d_0000 + n as u64 * 31);
        for bf16_input in [false, true] {
            let global = {
                let o = operands(&mut gpu, &case, G_KH, G_VH, bf16_input);
                run_chain(&mut gpu, &o, &case);
                let r = download(&mut gpu, &o, rows);
                free(&mut gpu, o);
                r
            };
            for rank in 0..2 {
                let local_in = local_case(&case, rank);
                let o = operands(&mut gpu, &local_in, L_KH, L_VH, bf16_input);
                run_chain(&mut gpu, &o, &local_in);
                let local = download(&mut gpu, &o, rows);
                free(&mut gpu, o);
                let what = if bf16_input {
                    "bf16 input"
                } else {
                    "f32 input"
                };
                let bytes = compare(what, rank, rows, &global, &local);
                eprintln!("{what}, rank {rank}, rows {rows:>5}: {bytes} bytes identical");
            }
        }
    }
}
