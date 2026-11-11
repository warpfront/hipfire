// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! `mq6_i8_gemm` workload: the builder-emitted MQ6G256V2 x A8 `block_i8_128`
//! INT8 prefill GEMM for gfx1201 (`peacemaker-isa emit --kernel mq6_i8_gemm`)
//! against the shipped object `kernels/gemm_mq6g256v2_mmq_i8_pm_gfx1201.hxaco`.
//!
//! # Shape grammar
//! `M=<rows>,N=<tokens>,K=<k>` (all three required, any order), `K % 256 == 0`,
//! `M % 4 == 0` (the PM entries store whole row quads, `mq6_i8_pm_admits`),
//! `N >= 1`; ragged M/N tiles are handled by the kernel. Or a preset
//! `fn:<name>`, the Fn gfx1201 prefill shapes
//! (N in {4096, 16384} x (M, K) in {(10240,2560), (6144,2560), (12288,2560),
//! (2560,6144), (512,2560), (128,2560)}):
//!
//! ```text
//! fn:m10240k2560-n4096   fn:m10240k2560-n16384
//! fn:m6144k2560-n4096    fn:m6144k2560-n16384
//! fn:m12288k2560-n4096   fn:m12288k2560-n16384
//! fn:m2560k6144-n4096    fn:m2560k6144-n16384
//! fn:m512k2560-n4096     fn:m512k2560-n16384
//! fn:m128k2560-n4096     fn:m128k2560-n16384
//! ```
//!
//! # Knobs (candidate; names as `peacemaker-isa emit --kernel mq6_i8_gemm`)
//! `epi` (`set` F32 | `set_bf16` BF16 bits), `raster` (`banded`, the product,
//! default | `linear`, the CPU-interpreter raster), `tile` (`128x128`, the
//! shipped tile, default | `256x128` | `128x256`: weight rows x tokens per
//! workgroup; the two wide tiles run 16 waves of the same 32 x 64 per-wave tile,
//! 55296 B of LDS). The launch of the `128x128` tile is exactly the runtime's
//! (`gemm_mq6g256v2_i8_prequant`, `mq6_i8_launch`): grid
//! `[ceil(M/128), ceil(N/128), 1]`, 256 threads, 36864 B dynamic LDS, kernarg
//! `A, Xq, Y` pointers then `M, K, N` and the PM image's seventh word (`0`); the
//! other tiles' grid, block and LDS follow the spec (`Tile::{rows, tokens,
//! threads, layout}`). `ring` (`2`, the shipped, default | `3`) is the A/W ring depth
//! (`Tune { ring }`, recorded in the candidate knobs only when non-default; the variant gains
//! `-r3`): depth 3 fits the `128x128` tile only (53248 B of LDS; the other tiles are refused with
//! the numbers) and keeps the depth-2 dynamic launch size, the third slab being the object's
//! static LDS; `base-ring` sets the ring depth of a `--base-tile` / `--base-rip` baseline. The baseline is always the shipped `128x128` object and
//! launches with its own geometry; the output is the same bits for every tile
//! (exact integer dot, the same per-element fold order), so
//! `identical_to_baseline` stays required.
//!
//! # Baseline
//! The shipped PM object's 128x128 entry (the pre-128x256 kernel, byte for byte; the
//! runtime now takes the `_t128x256` entries of the same object for wide launches,
//! `gemm_mq6_i8.rs` `mq6_i8_pm_token_tile`): `gemm_mq6g256v2_mmq_i8_pm_set_gfx1201` (F32) or
//! `..._pm_set_bf16_gfx1201` (BF16) (`gemm_mq6_i8.rs` `MQ6_I8_PM_ENTRY_*`; the
//! hipcc entry `mq6_i8_entry` is the non-admitted fallback and is not in this
//! object). Selection file:line and the object sha256 go in
//! `baseline.source`.
//!
//! `--base-shipped 128x256` (`128x128`, the default, | `128x256`) makes the baseline the shipped
//! object's `_t128x256` entry (launched with the 128x256 geometry) instead: the entry the runtime
//! takes for wide launches (`mq6_i8_pm_token_tile`). Use it when the runtime's selector picks the
//! wide tile at the shape, so the incumbent and the code-identical control are the product's
//! actual tile. Mutually exclusive with `--base-tile` / `--base-rip`.
//!
//! # Correctness
//! The candidate must be bitwise equal to the shipped baseline (the driver
//! requires `identical_to_baseline` for this workload) AND equal, bit for bit,
//! to the CPU reference (a transcription of the kernel's exact arithmetic,
//! `crates/peacemaker-isa/tests/mq6_i8_gemm.rs::reference`: per K128 block the
//! integer chain `C = sum q*qx`, `fc = float(C - 32*s)`, then
//! `acc = fma(RN(s_h*d), fc, acc); acc = fma(RN(z'_h*d), float(s), acc)` with
//! `z' = fma(32, s_h, z_h)`; BF16 = `isfinite ? RNE : truncate` of `0 + acc`).
//! The reference is exact; it is evaluated on all `(row, token)` pairs while
//! `M*N*K <= 2e9` MACs, else on a sampled grid (rows `0..64`, the last 64 and
//! `n` spread; the same for tokens): `--ref-rows` / `--ref-cols`
//! `full|sample:<n>` (default `sample:192`). The record names the covered rows
//! and tokens and the mismatch count.
//!
//! # Inputs (deterministic)
//! Weights in the exact MQ6G256V2 format (`mq6_fp8_gemm_bench.rs::gen_weights`
//! distribution: `s = f16(1e-4 + |N(0,1)|*1e-3)`, `z = f16(-32 s + N(0,1) s)`,
//! uniform 6-bit codes); activations N(0,1) with 0.1% x30 outliers quantized to
//! A8 blocks `{f32 d = amax/127, i32 s = sum q, i8 qs[128]}` laid `[K/128][N]`
//! (qs byte `j` holds natural K element `nat(j)`, the producer's nibble-pair
//! order).
//!
//! # `--emu`
//! Runs the candidate object on the peacemaker emulator at M=132,N=130,K=512 and
//! compares every output element with the reference. The emulator has no SALU
//! float-reciprocal semantics, so the banded raster is refused with the
//! emulator's own message: use `--raster linear`.

use crate::common::{contains, dev_err, sha256_hex};
use crate::iu4::{abi_signature, max_diff, sample_rows};
use crate::workload::{Arm, Knobs, Launch, Object, RefReport, Result, Workload};
use hipfire_kernel_abi::families::{find_export, mq6_i8 as mq6_abi};
use hipfire_kernel_abi::{
    check_export, eval, validate_invocation, Arch as AbiArch, ArgEvidence, Axis, Drift,
    EvidenceSource, ExportEvidence, HiddenKind, InvocationV1, KernelAbiV1, OperandBinding,
};
use peacemaker_isa::kernels::mq6_i8_gemm::{self, Epi, Raster, RingDepth, Spec, Tile, Tune};
use peacemaker_isa::Arch;
use rdna_compute::Gpu;
use serde_json::{json, Value};
use std::cell::OnceCell;
use std::path::Path;

const GROUP_BYTES: usize = 200;
const XBLK_BYTES: usize = 136;
const FULL_REF_MACS: u128 = 2_000_000_000;
const OBJECT: &str = "kernels/gemm_mq6g256v2_mmq_i8_pm_gfx1201.hxaco";

/// `(preset name, M, K)` of the Fn gfx1201 shapes, each at both N.
const FN_MK: [(&str, usize, usize); 6] = [
    ("m10240k2560", 10240, 2560),
    ("m6144k2560", 6144, 2560),
    ("m12288k2560", 12288, 2560),
    ("m2560k6144", 2560, 6144),
    ("m512k2560", 512, 2560),
    ("m128k2560", 128, 2560),
];
const FN_N: [usize; 2] = [4096, 16384];

/// Every `fn:<name>` preset with its `(M, N, K)`.
fn fn_presets() -> Vec<(String, (usize, usize, usize))> {
    FN_MK
        .iter()
        .flat_map(|&(name, m, k)| {
            FN_N.iter()
                .map(move |&n| (format!("{name}-n{n}"), (m, n, k)))
        })
        .collect()
}

fn parse_shape(s: &str) -> Result<(usize, usize, usize)> {
    if let Some(name) = s.strip_prefix("fn:") {
        return fn_presets()
            .into_iter()
            .find(|(n, _)| n == name)
            .map(|(_, mnk)| mnk)
            .ok_or_else(|| {
                format!(
                    "unknown preset fn:{name} (one of {})",
                    fn_presets()
                        .iter()
                        .map(|(n, _)| format!("fn:{n}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            });
    }
    let (mut m, mut n, mut k) = (None, None, None);
    for part in s.split(',') {
        let (key, v) = part
            .split_once('=')
            .ok_or_else(|| format!("shape `{s}`: expected M=..,N=..,K=.. or fn:<name>"))?;
        let v: usize = v
            .trim()
            .parse()
            .map_err(|e| format!("shape `{s}`: {key}: {e}"))?;
        let slot = match key.trim() {
            "M" => &mut m,
            "N" => &mut n,
            "K" => &mut k,
            other => return Err(format!("shape `{s}`: unknown dimension `{other}`")),
        };
        if slot.replace(v).is_some() {
            return Err(format!("shape `{s}`: duplicate {key}"));
        }
    }
    match (m, n, k) {
        (Some(m), Some(n), Some(k)) => Ok((m, n, k)),
        _ => Err(format!("shape `{s}`: M, N and K are all required")),
    }
}

/// xorshift64*.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    /// Uniform in `(0, 1]`.
    fn unit(&mut self) -> f64 {
        ((self.next() >> 11) + 1) as f64 / (1u64 << 53) as f64
    }
    /// Standard normal (Box-Muller, one variate per call).
    fn normal(&mut self) -> f64 {
        let (a, b) = (self.unit(), self.unit());
        (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
    }
}

fn f16_bits(x: f64) -> u16 {
    half::f16::from_f64(x).to_bits()
}

fn f16_val(b: u16) -> f32 {
    half::f16::from_bits(b).to_f32()
}

/// `m` rows x `k/256` MQ6G256V2 groups (the bench's distribution).
fn synth_weights(m: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut r = Rng::new(seed);
    let mut w = vec![0u8; m * (k / 256) * GROUP_BYTES];
    for g in w.chunks_exact_mut(GROUP_BYTES) {
        for h in 0..2 {
            let s = half::f16::from_f64(1e-4 + r.normal().abs() * 1e-3);
            let sf = f64::from(s.to_f32());
            g[4 * h..4 * h + 2].copy_from_slice(&s.to_bits().to_le_bytes());
            let z = f16_bits(-32.0 * sf + r.normal() * sf);
            g[4 * h + 2..4 * h + 4].copy_from_slice(&z.to_le_bytes());
        }
        for c in g[8..].chunks_mut(8) {
            let v = r.next().to_le_bytes();
            c.copy_from_slice(&v[..c.len()]);
        }
    }
    w
}

/// Natural K index (within the K128 block) of qs byte `j` (nibble-pair order).
fn nat(j: usize) -> usize {
    let (g, r) = (j / 8, j % 8);
    8 * g + if r < 4 { 2 * r } else { 2 * (r - 4) + 1 }
}

/// A8 blocks `[K/128][N]` from N(0,1) activations with 0.1% x30 outliers.
fn synth_x(n: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut r = Rng::new(seed);
    let mut x = vec![0u8; k / 128 * n * XBLK_BYTES];
    let mut v = [0f32; 128];
    // qs byte j holds natural element nat(j): the inverse permutation.
    let mut inv_nat = [0usize; 128];
    for j in 0..128 {
        inv_nat[nat(j)] = j;
    }
    for kb in 0..k / 128 {
        for t in 0..n {
            for e in &mut v {
                let g = r.normal();
                *e = (if r.next() % 1000 == 0 { g * 30.0 } else { g }) as f32;
            }
            let amax = v.iter().fold(0f32, |a, e| a.max(e.abs()));
            let d = amax / 127.0;
            let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
            let o = (kb * n + t) * XBLK_BYTES;
            let mut sum = 0i32;
            for (i, e) in v.iter().enumerate() {
                let q = (e * inv).round().clamp(-127.0, 127.0) as i32;
                sum += q;
                x[o + 8 + inv_nat[i]] = q as i8 as u8;
            }
            x[o..o + 4].copy_from_slice(&d.to_le_bytes());
            x[o + 4..o + 8].copy_from_slice(&sum.to_le_bytes());
        }
    }
    x
}

pub struct Host {
    /// `[M][K/256][200]`.
    w: Vec<u8>,
    /// `[K/128][N][136]`.
    x: Vec<u8>,
}

impl Host {
    fn synth(m: usize, n: usize, k: usize) -> Self {
        Self {
            w: synth_weights(m, k, 0x9e37_79b9_7f4a_7c15),
            x: synth_x(n, k, 0x1234_5678_9abc_def1),
        }
    }
}

/// 6-bit code `c` (natural index) of K128 half `half` of one group.
fn code(group: &[u8], half: usize, c: usize) -> i32 {
    let bits = 6 * c;
    let payload = &group[8 + 96 * half..8 + 96 * half + 96];
    let (byte, sh) = (bits / 8, bits % 8);
    let v = u32::from(payload[byte]) | u32::from(*payload.get(byte + 1).unwrap_or(&0)) << 8;
    ((v >> sh) & 63) as i32
}

/// Output element bits (F32 bits, or BF16 bits in the low 16) of `acc`.
fn out_bits(acc: f32, bf16: bool) -> u32 {
    if !bf16 {
        return acc.to_bits();
    }
    let y = 0.0f32 + acc;
    let u = y.to_bits();
    if y.is_finite() {
        u.wrapping_add(0x7fff + ((u >> 16) & 1)) >> 16
    } else {
        u >> 16
    }
}

/// Reference bits for `rows` x `cols` (row-major in `rows` order, `cols` within).
fn reference(
    host: &Host,
    m: usize,
    n: usize,
    k: usize,
    bf16: bool,
    rows: &[usize],
    cols: &[usize],
) -> Vec<u32> {
    let _ = m;
    let (gpr, nkb) = (k / 256, k / 128);
    let mut out = vec![0u32; rows.len() * cols.len()];
    let threads = std::thread::available_parallelism().map_or(1, |t| t.get());
    let per = rows.len().div_ceil(threads).max(1);
    std::thread::scope(|s| {
        for (rs, os) in rows.chunks(per).zip(out.chunks_mut(per * cols.len())) {
            s.spawn(move || {
                // codes[kb][j]: the code paired with qs byte j; (s_h, z'_h) per kb.
                let mut codes = vec![0i32; k];
                let mut sz = vec![(0f32, 0f32); nkb];
                for (&r, orow) in rs.iter().zip(os.chunks_exact_mut(cols.len())) {
                    for kb in 0..nkb {
                        let (gi, h) = (kb / 2, kb % 2);
                        let grp = &host.w[(r * gpr + gi) * GROUP_BYTES..][..GROUP_BYTES];
                        for j in 0..128 {
                            codes[kb * 128 + j] = code(grp, h, nat(j));
                        }
                        let sh = f16_val(u16::from_le_bytes([grp[4 * h], grp[4 * h + 1]]));
                        let zh = f16_val(u16::from_le_bytes([grp[4 * h + 2], grp[4 * h + 3]]));
                        sz[kb] = (sh, 32.0f32.mul_add(sh, zh));
                    }
                    for (&t, o) in cols.iter().zip(orow.iter_mut()) {
                        let mut acc = 0.0f32;
                        for kb in 0..nkb {
                            let xb = &host.x[(kb * n + t) * XBLK_BYTES..][..XBLK_BYTES];
                            let d = f32::from_le_bytes(xb[0..4].try_into().unwrap());
                            let sx = i32::from_le_bytes(xb[4..8].try_into().unwrap());
                            let cq: i32 = codes[kb * 128..][..128]
                                .iter()
                                .zip(&xb[8..])
                                .map(|(&q, &b)| q * i32::from(b as i8))
                                .sum();
                            let fc = (cq - 32 * sx) as f32;
                            let (sh, zp) = sz[kb];
                            acc = (sh * d).mul_add(fc, acc);
                            acc = (zp * d).mul_add(sx as f32, acc);
                        }
                        *o = out_bits(acc, bf16);
                    }
                }
            });
        }
    });
    out
}

/// The explicit kernarg of `spec`'s launch, encoded by the `hipfire-kernel-abi` contract of the
/// PM entry with the same epilogue (`families::mq6_i8`), plus that contract.
///
/// `ops` is `(address, allocation bytes)` of `A, Xq, Y`; `geom` the launch's real
/// `(grid, threads, dynamic LDS)`. A spec whose symbol is a contract export (the shipped
/// `128x128` / `t128x256` entries) is validated with its own geometry. Any other spec
/// (another tile or the linear raster) encodes through the same-epilogue product export, and its
/// generator kernarg layout must agree with the contract on every argument and the segment;
/// threads, LDS and symbol legitimately differ, anything else is an error before launch.
fn contract_kernarg(
    spec: Spec,
    (m, k, n): (usize, usize, usize),
    ops: [(u64, u64); 3],
    geom: ([u32; 3], u32, u32),
) -> Result<(Vec<u8>, &'static KernelAbiV1)> {
    let exact = if spec.epi == Epi::SetNr {
        None
    } else {
        find_export(AbiArch::Gfx1201, &spec.symbol())
    };
    let (abi, export) = match exact {
        Some(found) => found,
        None => {
            let product = Spec::product(if spec.epi == Epi::SetBf16 {
                Epi::SetBf16
            } else {
                Epi::Set
            });
            find_export(AbiArch::Gfx1201, &product.symbol())
                .ok_or_else(|| format!("no kernel-abi contract export for {}", product.symbol()))?
        }
    };
    let layout = spec.kernargs();
    let evidence = ExportEvidence {
        source: EvidenceSource::Generator,
        arch: Some(AbiArch::Gfx1201),
        symbol: export.symbol.to_owned(),
        args: layout
            .args
            .iter()
            .map(|a| ArgEvidence {
                name: a.name.clone(),
                offset: a.offset,
                size: a.size,
                value_kind: a.value_kind.clone(),
                access: a.actual_access.clone(),
            })
            .collect(),
        segment_bytes: Some(layout.size),
        segment_align: None,
        max_flat_workgroup_size: None,
        threads: None,
        dynamic_lds: None,
        group_segment_fixed: None,
        wave: None,
        object_sha256: None,
    };
    let drift: Vec<String> = check_export(abi, export, &evidence)
        .into_iter()
        .filter(|d| {
            matches!(
                d,
                Drift::MissingArg { .. }
                    | Drift::ExtraArg { .. }
                    | Drift::ArgName { .. }
                    | Drift::ArgWidth { .. }
                    | Drift::ArgKind { .. }
                    | Drift::ArgAccess { .. }
                    | Drift::Segment { .. }
                    | Drift::SegmentAlign { .. }
            )
        })
        .map(|d| d.to_string())
        .collect();
    if !drift.is_empty() {
        return Err(format!(
            "{}: kernarg layout differs from contract {} ({}): {}",
            spec.variant_name(),
            abi.id,
            export.symbol,
            drift.join("; ")
        ));
    }
    let params = [m as u64, k as u64, n as u64];
    let y_format = if spec.epi == Epi::SetBf16 {
        mq6_abi::FORMAT_BF16_N_M
    } else {
        mq6_abi::FORMAT_F32_N_M
    };
    let binds: [OperandBinding; 3] = [
        (mq6_abi::FORMAT_MQ6G256V2_WEIGHTS, ops[0]),
        (mq6_abi::FORMAT_BLOCK_I8_128_KB_N, ops[1]),
        (y_format, ops[2]),
    ]
    .map(|(format, (address, bytes))| OperandBinding {
        address,
        bytes: Some(bytes),
        format,
    });
    let (grid, block, dynamic_lds) = if exact.is_some() {
        (geom.0, [geom.1, 1, 1], geom.2)
    } else {
        let mut grid = [0u32; 3];
        for (g, e) in grid.iter_mut().zip(&export.grid) {
            let v = eval(e, abi, export, &params).map_err(|e| format!("{}: {e}", abi.id))?;
            *g = u32::try_from(v).map_err(|_| format!("{}: grid {v} exceeds u32", abi.id))?;
        }
        (grid, export.block, export.dynamic_lds)
    };
    let v = validate_invocation(
        abi,
        &InvocationV1 {
            arch: AbiArch::Gfx1201,
            symbol: export.symbol,
            params: &params,
            operands: &binds,
            grid,
            block,
            dynamic_lds,
        },
    )
    .map_err(|e| format!("{}: {e}", spec.variant_name()))?;
    Ok((v.explicit().to_vec(), abi))
}

/// `full` or `sample:<n>` indices over `0..dim`.
fn parse_axis(mode: &str, dim: usize, flag: &str) -> Result<(Vec<usize>, String)> {
    if mode == "full" {
        Ok(((0..dim).collect(), "full".to_owned()))
    } else if let Some(c) = mode.strip_prefix("sample:") {
        let c: usize = c.parse().map_err(|e| format!("--{flag} {mode}: {e}"))?;
        Ok((sample_rows(dim, c.max(2)), format!("sample:{c}")))
    } else {
        Err(format!("--{flag} {mode}: expected full|sample:<n>"))
    }
}

struct Dev {
    w: hip_bridge::DeviceBuffer,
    x: hip_bridge::DeviceBuffer,
    y: [hip_bridge::DeviceBuffer; 2],
}

pub struct Mq6 {
    m: usize,
    n: usize,
    k: usize,
    cand: Spec,
    /// `--rip`: source path and text of the candidate script.
    rip: Option<(String, String)>,
    /// `--base-tile` / `--base-rip`: the baseline is an emitted (or scripted) variant of this spec
    /// instead of the shipped object (hill-climbing against a tile's generator output).
    base: Option<(Spec, Option<(String, String)>)>,
    /// `--base-shipped`: the baseline is the shipped PM object's entry of this tile (the entry the
    /// runtime launches for the shape), not the `128x128` product entry.
    base_shipped: Option<Tile>,
    rows: Vec<usize>,
    rows_desc: String,
    cols: Vec<usize>,
    cols_desc: String,
    host: Option<Host>,
    dev: Option<Dev>,
    reference: OnceCell<Vec<u32>>,
}

/// The A/W ring depth an `mq6_i8_gemm` script declares (`const RING: u32 = N;`; `2` when absent).
fn script_ring(text: &str) -> Result<&'static str> {
    for line in text.lines() {
        if let Some(v) = line.trim().strip_prefix("const RING: u32 =") {
            return match v.trim().trim_end_matches(';').trim() {
                "2" => Ok("2"),
                "3" => Ok("3"),
                o => Err(format!("const RING = {o}: expected 2 or 3")),
            };
        }
    }
    Ok("2")
}
fn parse_spec(epi: &str, raster: &str, tile: &str, ring: &str) -> Result<Spec> {
    let epi: Epi = epi.parse()?;
    let tile: Tile = tile.parse().map_err(|e| format!("--tile: {e}"))?;
    let ring: RingDepth = ring.parse().map_err(|e| format!("--ring: {e}"))?;
    let raster = match raster {
        "banded" => Raster::Banded,
        "linear" => Raster::Linear,
        o => return Err(format!("--raster {o}: expected banded|linear")),
    };
    let spec = Spec {
        epi,
        raster,
        tile,
        arch: Arch::Gfx1201,
        tune: Tune { ring },
    };
    spec.validate()?;
    Ok(spec)
}

pub fn new(shape: &str, knobs: &mut Knobs, rip: Option<&Path>) -> Result<Box<dyn Workload>> {
    let (m, n, k) = parse_shape(shape)?;
    if m == 0 || n == 0 || k == 0 || k % 256 != 0 || m % 4 != 0 {
        return Err(format!(
            "shape M={m},N={n},K={k}: need M%4==0, K%256==0, N>=1"
        ));
    }
    let (epi_k, raster_k, tile_k) = (
        knobs.take_or("epi", "set"),
        knobs.take_or("raster", "banded"),
        knobs.take_or("tile", "128x128"),
    );
    let mut ring_k = knobs.take_or("ring", "2");
    // With `--rip` the ring depth is the script's own `const RING` (the twin has no ring knob);
    // a conflicting `--ring` is refused.
    if let Some(p) = rip {
        let text = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
        let script = script_ring(&text)?;
        if ring_k != "2" && ring_k != script {
            return Err(format!(
                "--ring {ring_k} conflicts with {}: its `const RING` is {script}",
                p.display()
            ));
        }
        ring_k = script.to_owned();
    }
    let cand = parse_spec(&epi_k, &raster_k, &tile_k, &ring_k)?;
    let (base_tile, base_rip) = (knobs.take("base-tile"), knobs.take("base-rip"));
    let base_shipped = match knobs.take("base-shipped") {
        None => None,
        Some(_) if base_tile.is_some() || base_rip.is_some() => {
            return Err(
                "--base-shipped selects a shipped entry; it excludes --base-tile/--base-rip".into(),
            )
        }
        Some(t) => match t
            .parse::<Tile>()
            .map_err(|e| format!("--base-shipped: {e}"))?
        {
            t @ (Tile::T128x128 | Tile::T128x256) => Some(t),
            _ => {
                return Err(format!(
                "--base-shipped {t}: the shipped object has the 128x128 and 128x256 entries only"
            ))
            }
        },
    };
    let base = if base_tile.is_some() || base_rip.is_some() {
        // The baseline's ring depth is its own (default 2): `--base-ring`.
        let base_ring = knobs.take_or("base-ring", "2");
        let spec = parse_spec(
            &epi_k,
            &raster_k,
            base_tile.as_deref().unwrap_or(&tile_k),
            &base_ring,
        )?;
        let rip = match &base_rip {
            Some(p) => Some((
                p.clone(),
                std::fs::read_to_string(p).map_err(|e| format!("{p}: {e}"))?,
            )),
            None => None,
        };
        Some((spec, rip))
    } else {
        None
    };
    let macs = (m as u128) * (n as u128) * (k as u128);
    let default = if macs <= FULL_REF_MACS {
        "full"
    } else {
        "sample:192"
    };
    let (rows, rows_desc) = parse_axis(&knobs.take_or("ref-rows", default), m, "ref-rows")?;
    let (cols, cols_desc) = parse_axis(&knobs.take_or("ref-cols", default), n, "ref-cols")?;
    let rip = match rip {
        Some(p) => Some((
            p.display().to_string(),
            std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?,
        )),
        None => None,
    };
    Ok(Box::new(Mq6 {
        m,
        n,
        k,
        cand,
        rip,
        base,
        base_shipped,
        rows,
        rows_desc,
        cols,
        cols_desc,
        host: None,
        dev: None,
        reference: OnceCell::new(),
    }))
}

/// Compile `.rip` source for `spec`, refuse unless its ABI equals that of
/// `mq6_i8_gemm::emit(spec)`; returns the ELF, symbol and the `candidate.rip`
/// fields (`variant`, `sha256`, `identical_to_generator`).
fn compile_rip(source: &str, spec: Spec) -> Result<(Vec<u8>, String, Value)> {
    // The variant name the script is built under: the ring depth is the script's own `const RING`
    // (the generator's `-r3` suffix is not part of the twin's variant grammar).
    let variant = spec.with_tune(Default::default()).variant_name();
    let compiled = peacemaker_rip::compile(source, spec.arch, &variant)?;
    let generator = mq6_i8_gemm::emit(spec)?;
    let symbol = spec.symbol();
    let elf = peacemaker_isa::native::assemble(&compiled.s_text, spec.arch)?;
    let gen_elf = peacemaker_isa::native::assemble(&generator.s_text, spec.arch)?;
    let got = abi_signature(&elf, &symbol)
        .map_err(|e| format!("compiled kernel does not expose {symbol}: {e}"))?;
    let want = abi_signature(&gen_elf, &symbol)?;
    if got != want {
        return Err(format!(
            "ABI mismatch with mq6_i8_gemm::emit({variant}):\n  .rip:      {got}\n  generator: {want}"
        ));
    }
    let info = json!({
        "sha256": sha256_hex(source.as_bytes()),
        "variant": variant,
        "identical_to_generator": compiled.s_text == generator.s_text,
    });
    Ok((elf, symbol, info))
}

/// `(grid, threads, dynamic LDS)` of `tile`'s launch at `M` rows and `N` tokens (depth-2 rings).
fn geometry(m: usize, n: usize, tile: Tile) -> ([u32; 3], u32, u32) {
    (
        [
            m.div_ceil(tile.rows() as usize) as u32,
            n.div_ceil(tile.tokens() as usize) as u32,
            1,
        ],
        tile.threads(),
        tile.layout().launch,
    )
}

/// [`geometry`] of a spec (`Spec::launch_lds`: the depth-2 dynamic size at either ring depth).
fn geometry_for(m: usize, n: usize, spec: Spec) -> ([u32; 3], u32, u32) {
    let (grid, threads, _) = geometry(m, n, spec.tile);
    (grid, threads, spec.launch_lds())
}

impl Mq6 {
    fn bf16(&self) -> bool {
        self.cand.epi == Epi::SetBf16
    }

    /// The spec `arm` launches: the shipped `128x128` product entry of the candidate's epilogue,
    /// or the emitted `--base-tile` / `--base-rip` spec.
    fn arm_spec(&self, arm: Arm) -> Spec {
        match arm {
            Arm::Baseline => self.base.as_ref().map_or_else(
                || Spec {
                    tile: self.base_shipped.unwrap_or(Tile::T128x128),
                    ..Spec::product(self.cand.epi)
                },
                |(s, _)| *s,
            ),
            Arm::Candidate => self.cand,
        }
    }

    /// `arm`'s explicit kernarg, from the contract, over the workload's live allocations.
    fn kernarg(&self, arm: Arm) -> Result<Vec<u8>> {
        let dev = self.dev.as_ref().ok_or("kernarg before setup")?;
        let y = match arm {
            Arm::Baseline => &dev.y[0],
            Arm::Candidate => &dev.y[1],
        };
        let spec = self.arm_spec(arm);
        let op = |b: &hip_bridge::DeviceBuffer| (b.as_ptr() as u64, b.size() as u64);
        contract_kernarg(
            spec,
            (self.m, self.k, self.n),
            [op(&dev.w), op(&dev.x), op(y)],
            geometry_for(self.m, self.n, spec),
        )
        .map(|(k, _)| k)
    }

    fn width(&self) -> usize {
        if self.bf16() {
            2
        } else {
            4
        }
    }

    fn out_bytes(&self) -> usize {
        self.n * self.m * self.width()
    }

    fn emitted(&self) -> Result<Object> {
        let e = mq6_i8_gemm::emit(self.cand)?;
        let elf = peacemaker_isa::native::assemble(&e.s_text, self.cand.arch)?;
        let symbol = self.cand.symbol();
        Ok(Object {
            source: json!({
                "source": "peacemaker_isa::kernels::mq6_i8_gemm::emit + native::assemble (peacemaker-isa emit --co)",
                "variant": self.cand.variant_name(),
                "sha256": sha256_hex(&elf),
            }),
            desc: json!({"knobs": self.knobs_json()}),
            exports: vec![symbol.clone()],
            launch_symbol: symbol,
            bytes: elf,
        })
    }

    fn rip_object(&self, path: &str, text: &str) -> Result<Object> {
        let (elf, symbol, mut info) = compile_rip(text, self.cand)?;
        info["path"] = json!(path);
        Ok(Object {
            source: json!({
                "source": format!("peacemaker_rip::compile({path}) + native::assemble"),
                "variant": self.cand.variant_name(),
                "sha256": sha256_hex(&elf),
            }),
            desc: json!({"knobs": self.knobs_json(), "rip": info}),
            exports: vec![symbol.clone()],
            launch_symbol: symbol,
            bytes: elf,
        })
    }

    fn knobs_json(&self) -> Value {
        let mut v = json!({"epi": self.cand.epi.name(), "raster": match self.cand.raster {
            Raster::Banded => "banded",
            Raster::Linear => "linear",
        }, "tile": self.cand.tile.name()});
        // Only non-default generator knobs are recorded: every existing record keeps its bytes.
        if self.cand.tune.ring != RingDepth::Two {
            v["ring"] = json!(self.cand.tune.ring.name());
        }
        v
    }

    /// Run the candidate object on the emulator at a small shape and compare
    /// every element with the reference.
    fn run_emu(&self) -> Result<Value> {
        use peacemaker_ir::emu::{Emulator, Launch as ELaunch, Memory};
        let obj = self.candidate()?;
        let lifted = peacemaker_lift::lift_object(
            &obj.bytes,
            peacemaker_lift::Options {
                frontend: peacemaker_ir::inst::Frontend::Builder,
            },
        )
        .map_err(|e| format!("lift: {e:?}"))?;
        let emu = Emulator::new(&lifted.program, &obj.launch_symbol)?.hw_model();
        let (m, n, k) = (132usize, 130usize, 512usize);
        let host = Host::synth(m, n, k);
        let (kva, ava, xva, yva) = (
            0x1000_0000u64,
            0x2000_0000u64,
            0x3000_0000u64,
            0x4000_0000u64,
        );
        let tile = self.cand.tile;
        let (gx, gy) = (
            m.div_ceil(tile.rows() as usize),
            n.div_ceil(tile.tokens() as usize),
        );
        let (mut kern, abi) = contract_kernarg(
            self.cand,
            (m, k, n),
            [
                (ava, host.w.len() as u64),
                (xva, host.x.len() as u64),
                (yva, (n * m * self.width()) as u64),
            ],
            geometry_for(m, n, self.cand),
        )?;
        kern.resize(self.cand.kernargs().size as usize, 0);
        for h in abi.hidden {
            if let HiddenKind::BlockCount(axis) = h.kind {
                let v = match axis {
                    Axis::X => gx as u32,
                    Axis::Y => gy as u32,
                    Axis::Z => 1,
                };
                let off = h.offset as usize;
                kern[off..off + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        let mut mem = Memory::default();
        mem.map("kernarg".into(), kva, kern)?;
        let ylen = n * m * self.width();
        for (name, va, bytes) in [
            ("a", ava, host.w.clone()),
            ("xq", xva, host.x.clone()),
            ("y", yva, vec![0; ylen]),
        ] {
            let len = bytes.len() as u64;
            mem.map(name.into(), va, bytes)?;
            mem.map(format!("{name}_tail"), va + len, vec![0; 4096])?;
        }
        for y in 0..gy {
            for x in 0..gx {
                emu.run_workgroup(
                    &ELaunch {
                        kernarg_ptr: kva,
                        workgroup: [x as u32, y as u32, 0],
                        block: [tile.threads(), 1, 1],
                        dynamic_lds: self.cand.launch_lds(),
                    },
                    &mut mem,
                )?;
            }
        }
        let got = mem.read(yva, ylen)?;
        let rows: Vec<usize> = (0..m).collect();
        let cols: Vec<usize> = (0..n).collect();
        let want = reference(&host, m, n, k, self.bf16(), &rows, &cols);
        let w = self.width();
        let mismatches = (0..m)
            .flat_map(|r| (0..n).map(move |c| (r, c)))
            .filter(|&(r, c)| {
                let o = (c * m + r) * w;
                let bits = if w == 2 {
                    u32::from(u16::from_le_bytes([got[o], got[o + 1]]))
                } else {
                    u32::from_le_bytes(got[o..o + 4].try_into().unwrap())
                };
                bits != want[r * n + c]
            })
            .count();
        Ok(json!({
            "ran": true,
            "shape": {"M": m, "N": n, "K": k},
            "symbol": obj.launch_symbol,
            "reference": "cpu_mq6_i8_fold",
            "mismatching_elements": mismatches,
            "bit_exact": mismatches == 0,
        }))
    }
}

impl Workload for Mq6 {
    fn kernel(&self) -> &'static str {
        "mq6_i8_gemm"
    }
    fn arch(&self) -> &'static str {
        "gfx1201"
    }
    fn shape_json(&self) -> Value {
        json!({"M": self.m, "N": self.n, "K": self.k})
    }
    fn candidate_knobs(&self) -> Value {
        let mut v = self.knobs_json();
        v["variant"] = json!(self.cand.variant_name());
        v
    }
    fn requires_baseline_identity(&self) -> bool {
        true
    }

    fn candidate(&self) -> Result<Object> {
        match &self.rip {
            Some((path, text)) => self.rip_object(path, text),
            None => self.emitted(),
        }
    }

    fn baseline(&self) -> Result<Object> {
        if let Some((spec, rip)) = &self.base {
            let (elf, symbol) = match rip {
                Some((path, text)) => {
                    let (elf, symbol, _) = compile_rip(text, *spec)?;
                    let _ = path;
                    (elf, symbol)
                }
                None => {
                    let e = mq6_i8_gemm::emit(*spec)?;
                    (
                        peacemaker_isa::native::assemble(&e.s_text, spec.arch)?,
                        spec.symbol(),
                    )
                }
            };
            return Ok(Object {
                source: json!({
                    "source": "--base-tile/--base-rip: second emitted (or scripted) variant, not the shipped object",
                    "variant": spec.variant_name(),
                    "base_rip": rip.as_ref().map(|(p, _)| p.clone()),
                    "sha256": sha256_hex(&elf),
                }),
                desc: json!({"variant": spec.variant_name()}),
                exports: vec![symbol.clone()],
                launch_symbol: symbol,
                bytes: elf,
            });
        }
        let bytes = rdna_compute::kernels::GEMM_MQ6G256V2_MMQ_I8_PM_GFX1201;
        if self.cand.epi == Epi::SetNr {
            return Err(
                "epi set_nr takes the QSA norm and RoPE-table kernel arguments, which this \
                 harness does not pass: tests/qsa_norm_epi_oracle.rs checks and times it"
                    .into(),
            );
        }
        // The shipped entry of the baseline's tile: the product symbol (128x128) or its
        // `_t128x256` sibling (`Spec::symbol` of the banded spec), the runtime's two PM entries.
        let symbol = self.arm_spec(Arm::Baseline).symbol();
        let symbol = symbol.as_str();
        if !contains(bytes, symbol.as_bytes()) {
            return Err(format!(
                "symbol {symbol} is not in the shipped object {OBJECT}"
            ));
        }
        Ok(Object {
            bytes: bytes.to_vec(),
            launch_symbol: symbol.to_owned(),
            exports: vec![symbol.to_owned()],
            desc: json!({"artifact": OBJECT, "knobs": {"epi": self.cand.epi.name(), "raster": "banded", "tile": self.arm_spec(Arm::Baseline).tile.name()}}),
            source: json!({
                "source": "crates/rdna-compute/src/gemm_mq6_i8.rs MQ6_I8_PM_MODULE / MQ6_I8_PM_ENTRY_F32 / MQ6_I8_PM_ENTRY_BF16 (the 128x128 entries) or, with --base-shipped 128x256, the _T256 entries; \
                           entry choice mq6_i8_pm_launch / mq6_i8_pm_token_tile and embedded-module load (gemm_mq6g256v2_i8_prequant), \
                           admission mq6_i8_pm_admits, 128x128 launch (grid [ceil(M/128), ceil(N/128)], 256 threads, 36864 B LDS)",
                "artifact": OBJECT,
                "kernels_const": "crates/rdna-compute/src/kernels.rs:4434 GEMM_MQ6G256V2_MMQ_I8_PM_GFX1201",
                "sha256": sha256_hex(bytes),
            }),
        })
    }

    fn setup(&mut self, gpu: &Gpu) -> Result<()> {
        let host = Host::synth(self.m, self.n, self.k);
        let up = |b: &[u8]| -> Result<hip_bridge::DeviceBuffer> {
            let d = gpu.hip.malloc(b.len().max(16)).map_err(dev_err)?;
            gpu.hip.memcpy_htod(&d, b).map_err(dev_err)?;
            Ok(d)
        };
        let out = self.out_bytes().max(16);
        let dev = Dev {
            w: up(&host.w)?,
            x: up(&host.x)?,
            y: [
                gpu.hip.malloc(out).map_err(dev_err)?,
                gpu.hip.malloc(out).map_err(dev_err)?,
            ],
        };
        self.host = Some(host);
        self.dev = Some(dev);
        // Refuse a wrong layout / extent / alignment before any launch.
        for arm in [Arm::Baseline, Arm::Candidate] {
            self.kernarg(arm)?;
        }
        for arm in [Arm::Baseline, Arm::Candidate] {
            self.reset(gpu, arm)?;
        }
        Ok(())
    }

    fn launch(&self, arm: Arm) -> Launch {
        // Each arm's grid, block and LDS follow its own tile (see `arm_spec`).
        let spec = self.arm_spec(arm);
        let (grid, threads, lds) = geometry_for(self.m, self.n, spec);
        Launch {
            kernarg: self
                .kernarg(arm)
                .expect("kernarg contract is validated in setup"),
            grid,
            block: [threads, 1, 1],
            lds,
        }
    }

    fn reset(&self, gpu: &Gpu, arm: Arm) -> Result<()> {
        let dev = self.dev.as_ref().ok_or("reset before setup")?;
        let y = match arm {
            Arm::Baseline => &dev.y[0],
            Arm::Candidate => &dev.y[1],
        };
        gpu.hip.memset(y, 0, y.size()).map_err(dev_err)
    }

    fn output(&self, gpu: &Gpu, arm: Arm) -> Result<Vec<u8>> {
        let dev = self.dev.as_ref().ok_or("output before setup")?;
        let y = match arm {
            Arm::Baseline => &dev.y[0],
            Arm::Candidate => &dev.y[1],
        };
        let mut v = vec![0u8; y.size()];
        gpu.hip.memcpy_dtoh(&mut v, y).map_err(dev_err)?;
        v.truncate(self.out_bytes());
        Ok(v)
    }

    fn check(&self, out: &[u8]) -> Result<RefReport> {
        if out.len() != self.out_bytes() {
            return Err(format!(
                "output is {} bytes, expected {}",
                out.len(),
                self.out_bytes()
            ));
        }
        let host = self.host.as_ref().ok_or("check before setup")?;
        let want = self.reference.get_or_init(|| {
            reference(
                host,
                self.m,
                self.n,
                self.k,
                self.bf16(),
                &self.rows,
                &self.cols,
            )
        });
        let w = self.width();
        let (mut mismatches, mut worst) = (0usize, 0f64);
        for (ri, &r) in self.rows.iter().enumerate() {
            for (ci, &t) in self.cols.iter().enumerate() {
                let o = (t * self.m + r) * w;
                let bits = if w == 2 {
                    u32::from(u16::from_le_bytes([out[o], out[o + 1]]))
                } else {
                    u32::from_le_bytes(out[o..o + 4].try_into().unwrap())
                };
                let exp = want[ri * self.cols.len() + ci];
                if bits != exp {
                    mismatches += 1;
                    worst = worst.max(max_diff(&out[o..o + w], &exp_bytes(exp, w), w));
                }
            }
        }
        Ok(RefReport {
            reference: format!(
                "cpu_mq6_i8_fold(rows={},tokens={},elements={},mismatching_elements={mismatches})",
                self.rows_desc,
                self.cols_desc,
                self.rows.len() * self.cols.len()
            ),
            max_abs_err: worst,
            tolerance: 0.0,
            pass: mismatches == 0,
        })
    }

    fn max_abs_diff(&self, a: &[u8], b: &[u8]) -> Result<f64> {
        if a.len() != b.len() {
            return Err(format!("length mismatch {} vs {}", a.len(), b.len()));
        }
        Ok(max_diff(a, b, self.width()))
    }

    fn emu(&self) -> Value {
        match self.run_emu() {
            Ok(v) => v,
            Err(e) => Value::String(format!("refused ({e})")),
        }
    }
}

fn exp_bytes(bits: u32, w: usize) -> Vec<u8> {
    if w == 2 {
        (bits as u16).to_le_bytes().to_vec()
    } else {
        bits.to_le_bytes().to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn knobs(kv: &[(&str, &str)]) -> Knobs {
        Knobs(kv.iter().map(|(k, v)| ((*k).into(), (*v).into())).collect())
    }

    #[test]
    fn shape_grammar_and_presets() {
        assert_eq!(
            parse_shape("M=12288,N=16384,K=2560").unwrap(),
            (12288, 16384, 2560)
        );
        assert_eq!(parse_shape("K=256,N=1,M=4").unwrap(), (4, 1, 256));
        assert!(parse_shape("M=4,N=1").is_err());
        assert!(parse_shape("M=4,N=1,K=256,K=256").is_err());
        assert!(parse_shape("fn:nope").is_err());
        assert!(new("M=4,N=1,K=128", &mut knobs(&[]), None).is_err());
        assert!(new("M=6,N=1,K=256", &mut knobs(&[]), None).is_err());
    }

    #[test]
    fn fn_preset_table_is_the_twelve_fn_shapes() {
        let p = fn_presets();
        assert_eq!(p.len(), 12);
        let mut mnk: Vec<_> = p.iter().map(|(_, s)| *s).collect();
        mnk.sort_unstable();
        mnk.dedup();
        assert_eq!(mnk.len(), 12, "every preset is a distinct shape");
        for (mk, n) in FN_MK
            .iter()
            .flat_map(|mk| FN_N.iter().map(move |n| (mk, n)))
        {
            assert!(mnk.contains(&(mk.1, *n, mk.2)), "{mk:?} N={n}");
        }
        assert_eq!(
            parse_shape("fn:m12288k2560-n16384").unwrap(),
            (12288, 16384, 2560)
        );
        assert_eq!(
            parse_shape("fn:m2560k6144-n4096").unwrap(),
            (2560, 4096, 6144)
        );
        for (name, (m, _, k)) in &p {
            assert!(
                m % 4 == 0 && k % 256 == 0,
                "{name} must satisfy the PM gate"
            );
        }
    }

    #[test]
    fn knob_to_spec_and_symbol() {
        let w = new(
            "fn:m128k2560-n4096",
            &mut knobs(&[("epi", "set_bf16"), ("raster", "linear")]),
            None,
        )
        .unwrap();
        assert_eq!(w.kernel(), "mq6_i8_gemm");
        assert_eq!(w.candidate_knobs()["variant"], "128x128x8-set_bf16-linear");
        assert!(w.requires_baseline_identity());
        let d = new("M=128,N=128,K=256", &mut knobs(&[]), None).unwrap();
        assert_eq!(d.candidate_knobs()["variant"], "128x128x8-set");
        assert!(new("M=128,N=128,K=256", &mut knobs(&[("epi", "add")]), None).is_err());
        assert!(new("M=128,N=128,K=256", &mut knobs(&[("raster", "x")]), None).is_err());
        assert!(new(
            "M=128,N=128,K=256",
            &mut knobs(&[("ref-rows", "some")]),
            None
        )
        .is_err());
    }

    #[test]
    fn tile_knob_selects_the_tile_and_the_launch_follows_it() {
        for (knob, variant, grid, block, lds) in [
            ("128x128", "128x128x8-set", [3u32, 2, 1], 256u32, 36864u32),
            ("256x128", "256x128x16-set", [2, 2, 1], 512, 55296),
            ("128x256", "128x256x16-set", [3, 1, 1], 512, 55296),
        ] {
            let w = new("M=260,N=130,K=256", &mut knobs(&[("tile", knob)]), None).unwrap();
            assert_eq!(w.candidate_knobs()["variant"], variant);
            assert_eq!(
                w.candidate_knobs()["tile"],
                variant.split('-').next().unwrap()
            );
            let c = w.candidate().unwrap();
            assert!(c
                .launch_symbol
                .starts_with("gemm_mq6g256v2_mmq_i8_pm_set_gfx1201"));
            // The baseline is the shipped object whatever the tile.
            assert_eq!(
                w.baseline().unwrap().launch_symbol,
                "gemm_mq6g256v2_mmq_i8_pm_set_gfx1201"
            );
            let tile: Tile = knob.parse().unwrap();
            assert_eq!(geometry(260, 130, tile), (grid, block, lds), "{knob}");
        }
        assert!(new("M=128,N=128,K=256", &mut knobs(&[("tile", "64x64")]), None).is_err());
    }

    #[test]
    fn baseline_is_the_shipped_pm_entry() {
        for (epi, sym) in [
            ("set", "gemm_mq6g256v2_mmq_i8_pm_set_gfx1201"),
            ("set_bf16", "gemm_mq6g256v2_mmq_i8_pm_set_bf16_gfx1201"),
        ] {
            let w = new("M=128,N=128,K=256", &mut knobs(&[("epi", epi)]), None).unwrap();
            let b = w.baseline().unwrap();
            assert_eq!(b.launch_symbol, sym);
            assert_eq!(b.source["artifact"], OBJECT);
            assert_eq!(b.source["sha256"].as_str().unwrap().len(), 64);
        }
    }

    #[test]
    fn base_shipped_selects_the_shipped_wide_entry() {
        for (epi, sym) in [
            ("set", "gemm_mq6g256v2_mmq_i8_pm_set_gfx1201_t128x256"),
            (
                "set_bf16",
                "gemm_mq6g256v2_mmq_i8_pm_set_bf16_gfx1201_t128x256",
            ),
        ] {
            let w = new(
                "M=10240,N=4096,K=2560",
                &mut knobs(&[
                    ("epi", epi),
                    ("tile", "128x256"),
                    ("base-shipped", "128x256"),
                ]),
                None,
            )
            .unwrap();
            let b = w.baseline().unwrap();
            assert_eq!(b.launch_symbol, sym);
            assert_eq!(b.source["artifact"], OBJECT);
            // The generator twin of the same tile launches the same symbol.
            assert_eq!(w.candidate().unwrap().launch_symbol, sym);
        }
        let w = new(
            "M=128,N=128,K=256",
            &mut knobs(&[("base-shipped", "128x128")]),
            None,
        )
        .unwrap();
        assert_eq!(
            w.baseline().unwrap().launch_symbol,
            "gemm_mq6g256v2_mmq_i8_pm_set_gfx1201"
        );
        for bad in [
            &[("base-shipped", "256x128")][..],
            &[("base-shipped", "64x64")],
            &[("base-shipped", "128x256"), ("base-tile", "128x128")],
        ] {
            assert!(new("M=128,N=128,K=256", &mut knobs(bad), None).is_err());
        }
    }

    #[test]
    fn emitted_candidate_symbol_follows_spec() {
        let w = new(
            "M=128,N=128,K=256",
            &mut knobs(&[("raster", "linear")]),
            None,
        )
        .unwrap();
        let c = w.candidate().unwrap();
        assert_eq!(
            c.launch_symbol,
            "gemm_mq6g256v2_mmq_i8_pm_set_gfx1201_linear"
        );
    }

    #[test]
    fn launch_is_the_runtime_geometry() {
        let (b, abi) = contract_kernarg(
            Spec::product(Epi::Set),
            (260, 256, 130),
            [(0x1000, 52000), (0x2000, 35360), (0x3000, 135200)],
            geometry(260, 130, Tile::T128x128),
        )
        .unwrap();
        assert_eq!(abi.id, "mq6_i8_gemm.pm.set_f32");
        assert_eq!(b.len(), 40);
        assert_eq!(u64::from_le_bytes(b[16..24].try_into().unwrap()), 0x3000);
        for (off, v) in [(24, 260), (28, 256), (32, 130), (36, 0)] {
            assert_eq!(i32::from_le_bytes(b[off..off + 4].try_into().unwrap()), v);
        }
    }

    /// An independent contract encoding (the explicit statics, `validate_invocation` +
    /// `encode_into`) equals the workload's kernarg for canonical shapes, tiles and epilogues.
    #[test]
    fn mq6_kernarg_is_contract_encoding() {
        use hipfire_kernel_abi::families::mq6_i8 as f;
        use hipfire_kernel_abi::{encode_into, validate_invocation};
        for (epi, abi, y_fmt, y_bytes_per) in [
            (Epi::Set, &f::MQ6_I8_PM_SET_F32, f::FORMAT_F32_N_M, 4u64),
            (
                Epi::SetBf16,
                &f::MQ6_I8_PM_SET_BF16,
                f::FORMAT_BF16_N_M,
                2u64,
            ),
        ] {
            for (m, n, k) in [(260usize, 130usize, 256usize), (128, 128, 512), (4, 1, 256)] {
                let ops = [
                    (0x1000u64, (m * (k / 256) * 200) as u64),
                    (0x2000, (n * (k / 128) * 136) as u64),
                    (0x3000, n as u64 * m as u64 * y_bytes_per),
                ];
                let spec = Spec::product(epi);
                let (got, got_abi) =
                    contract_kernarg(spec, (m, k, n), ops, geometry(m, n, Tile::T128x128)).unwrap();
                assert_eq!(got_abi.id, abi.id);
                let formats = [
                    f::FORMAT_MQ6G256V2_WEIGHTS,
                    f::FORMAT_BLOCK_I8_128_KB_N,
                    y_fmt,
                ];
                let binds: Vec<OperandBinding> = ops
                    .iter()
                    .zip(formats)
                    .map(|(&(address, bytes), format)| OperandBinding {
                        address,
                        bytes: Some(bytes),
                        format,
                    })
                    .collect();
                let (grid, threads, lds) = geometry(m, n, Tile::T128x128);
                let v = validate_invocation(
                    abi,
                    &InvocationV1 {
                        arch: AbiArch::Gfx1201,
                        symbol: &spec.symbol(),
                        params: &[m as u64, k as u64, n as u64],
                        operands: &binds,
                        grid,
                        block: [threads, 1, 1],
                        dynamic_lds: lds,
                    },
                )
                .unwrap();
                let mut want = vec![0u8; abi.explicit_bytes as usize];
                encode_into(&v, &mut want).unwrap();
                assert_eq!(got, want, "{} M={m} N={n} K={k}", epi.name());
                // Non-product tiles/rasters of the same epilogue share the product's image.
                for (tile, raster) in [
                    (Tile::T128x256, Raster::Banded),
                    (Tile::T256x128, Raster::Banded),
                    (Tile::T128x128, Raster::Linear),
                ] {
                    let s = Spec {
                        epi,
                        raster,
                        tile,
                        arch: Arch::Gfx1201,
                        tune: Default::default(),
                    };
                    let (g, _) = contract_kernarg(s, (m, k, n), ops, geometry(m, n, tile)).unwrap();
                    assert_eq!(g, want, "{} M={m} N={n} K={k}", s.variant_name());
                }
            }
        }
    }

    /// A candidate whose generator kernarg layout differs from the contract (set_nr appends the
    /// norm/RoPE pointers, renames word 36 and grows the segment) is refused before launch.
    #[test]
    fn mq6_wrong_candidate_layout_is_rejected_before_launch() {
        let nr = Spec {
            epi: Epi::SetNr,
            raster: Raster::Banded,
            tile: Tile::T256x128,
            arch: Arch::Gfx1201,
            tune: Default::default(),
        };
        let e = contract_kernarg(
            nr,
            (256, 256, 128),
            [(0x1000, 51200), (0x2000, 34816), (0x3000, 131072)],
            geometry(256, 128, Tile::T256x128),
        )
        .unwrap_err();
        assert!(e.contains("kernarg layout"), "{e}");
        assert!(e.contains("head_mask"), "{e}");
    }

    #[test]
    fn mq6_contract_refuses_short_extent_and_misalignment() {
        let g = geometry(260, 130, Tile::T128x128);
        let s = Spec::product(Epi::Set);
        let good = [(0x1000u64, 52000u64), (0x2000, 35360), (0x3000, 135200)];
        assert!(contract_kernarg(s, (260, 256, 130), good, g).is_ok());
        let mut short = good;
        short[2].1 -= 4;
        assert!(contract_kernarg(s, (260, 256, 130), short, g).is_err());
        let mut odd = good;
        odd[0].0 = 0x1001;
        assert!(contract_kernarg(s, (260, 256, 130), odd, g).is_err());
    }

    #[test]
    fn reference_hand_case() {
        // One row, one token, K=256: all codes 32 (q-32 = 0) and qx = 1 => C part 0,
        // acc = fma(RN(z'*d), s, 0) with z' = fma(32, s_h, z_h) = z_h + 32 s_h.
        let (m, n, k) = (4usize, 1usize, 256usize);
        let mut w = vec![0u8; m * 200];
        for r in 0..m {
            let g = &mut w[r * 200..(r + 1) * 200];
            for h in 0..2 {
                g[4 * h..4 * h + 2].copy_from_slice(&f16_bits(0.5).to_le_bytes());
                g[4 * h + 2..4 * h + 4].copy_from_slice(&f16_bits(-16.0).to_le_bytes());
            }
            // 6-bit code 32 = 0b100000 everywhere: bit 5 of every 6-bit field.
            for half in 0..2 {
                let p = &mut g[8 + 96 * half..8 + 96 * half + 96];
                for c in 0..128 {
                    let bit = 6 * c + 5;
                    p[bit / 8] |= 1 << (bit % 8);
                }
            }
        }
        let mut x = vec![0u8; 2 * XBLK_BYTES];
        for kb in 0..2 {
            let o = kb * XBLK_BYTES;
            x[o..o + 4].copy_from_slice(&2.0f32.to_le_bytes());
            x[o + 4..o + 8].copy_from_slice(&128i32.to_le_bytes());
            for j in 0..128 {
                x[o + 8 + j] = 1;
            }
        }
        let host = Host { w, x };
        assert_eq!(code(&host.w[0..200], 1, 127), 32);
        let got = reference(&host, m, n, k, false, &[0, 3], &[0]);
        // z' = -16 + 32*0.5 = 0, so every block contributes 0*d*128 = 0.
        assert_eq!(got, vec![0f32.to_bits(); 2]);
        // bf16 of 1.0 is 0x3f80.
        assert_eq!(out_bits(1.0, true), 0x3f80);
        assert_eq!(out_bits(f32::INFINITY, true), 0x7f80);
    }

    #[test]
    fn nat_is_a_permutation_and_synthesis_is_deterministic() {
        let mut seen = [false; 128];
        for j in 0..128 {
            seen[nat(j)] = true;
        }
        assert!(seen.iter().all(|&s| s));
        let a = Host::synth(8, 4, 256);
        let b = Host::synth(8, 4, 256);
        assert_eq!(a.w, b.w);
        assert_eq!(a.x, b.x);
        assert_eq!(a.w.len(), 8 * 200);
        assert_eq!(a.x.len(), 2 * 4 * XBLK_BYTES);
    }

    const TWIN: &str = include_str!("../../../peacemaker-isa/src/kernels/mq6_i8_gemm.rip");

    #[test]
    fn rip_twin_is_identical_to_generator() {
        for (epi, raster, tile) in [
            ("set", "banded", "128x128"),
            ("set_bf16", "banded", "128x128"),
            ("set", "linear", "128x128"),
            ("set_bf16", "linear", "128x128"),
            ("set", "banded", "256x128"),
            ("set_bf16", "linear", "256x128"),
            ("set", "banded", "128x256"),
            ("set_bf16", "linear", "128x256"),
        ] {
            let spec = parse_spec(epi, raster, tile, "2").unwrap();
            let (_, symbol, info) = compile_rip(TWIN, spec).unwrap();
            assert_eq!(symbol, spec.symbol());
            assert_eq!(
                info["identical_to_generator"], true,
                "{epi} {raster} {tile}"
            );
            assert_eq!(info["variant"], spec.variant_name());
            assert_eq!(info["sha256"], sha256_hex(TWIN.as_bytes()));
        }
    }

    #[test]
    fn rip_garbage_is_refused() {
        let spec = parse_spec("set", "banded", "128x128", "2").unwrap();
        assert!(compile_rip("fn emit(", spec).is_err());
    }

    /// `--ring 3`: the candidate is the depth-3 kernel (`-r3` variant, the knob recorded only when
    /// non-default), its launch carries the third A/W slab's LDS, and the shipped baseline stays
    /// depth 2.
    #[test]
    fn ring_knob_selects_depth_3_and_the_launch_follows_it() {
        let d2 = new("M=256,N=256,K=768", &mut knobs(&[]), None).unwrap();
        assert!(d2.candidate_knobs().get("ring").is_none());
        assert_eq!(d2.candidate_knobs()["variant"], "128x128x8-set");
        let d3 = new("M=256,N=256,K=768", &mut knobs(&[("ring", "3")]), None).unwrap();
        assert_eq!(d3.candidate_knobs()["ring"], "3");
        assert_eq!(d3.candidate_knobs()["variant"], "128x128x8-set-r3");
        let spec = parse_spec("set", "banded", "128x128", "3").unwrap();
        // The depth-3 object keeps the depth-2 dynamic launch size (the third slab is static LDS).
        assert_eq!(geometry_for(256, 256, spec).2, 36864);
        assert_eq!(spec.static_lds() + spec.launch_lds(), 53248);
        assert_eq!(geometry_for(256, 256, spec).0, [2, 2, 1]);
        assert_eq!(
            geometry_for(
                256,
                256,
                parse_spec("set", "banded", "128x128", "2").unwrap()
            )
            .2,
            36864
        );
        // The emitted candidate object is the depth-3 kernel (the dynamic LDS it launches with).
        let obj = d3.candidate().unwrap();
        assert_eq!(obj.launch_symbol, spec.symbol());
        assert_eq!(
            d3.baseline().unwrap().launch_symbol,
            "gemm_mq6g256v2_mmq_i8_pm_set_gfx1201"
        );
        assert!(new("M=256,N=256,K=768", &mut knobs(&[("ring", "4")]), None).is_err());
    }

    /// The wide tiles do not fit depth 3 (the third A+W slab takes the launch past 64 KiB, the
    /// second prefetch set past the 192-VGPR plan): the refusal names the numbers.
    #[test]
    fn ring_3_is_refused_on_the_wide_tiles_with_the_numbers() {
        for tile in ["128x256", "256x128"] {
            let e = new(
                "M=10240,N=4096,K=2560",
                &mut knobs(&[("tile", tile), ("ring", "3")]),
                None,
            )
            .err()
            .expect(tile);
            assert!(e.contains("ring depth 3 does not fit"), "{e}");
            assert!(e.contains("79872 B") && e.contains("65536 B"), "{e}");
        }
    }

    /// `--rip` of a `const RING = 3` copy of the twin at the FN prefill node's shape on the 128x128
    /// tile: the ring depth comes from the script (no `--ring`), the candidate object is the
    /// script's, identical to the depth-3 generator; a conflicting `--ring` is refused.
    #[test]
    fn rip_flag_reads_the_ring_from_the_script_at_the_fn_node_shape() {
        let dir = std::env::temp_dir().join(format!("pm-loop-mq6-rip-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ring3.rip");
        std::fs::write(
            &path,
            TWIN.replace("const RING: u32 = 2;", "const RING: u32 = 3;"),
        )
        .unwrap();
        let shape = "M=10240,N=4096,K=2560";
        let w = new(
            shape,
            &mut knobs(&[("tile", "128x128"), ("base-shipped", "128x256")]),
            Some(&path),
        )
        .unwrap();
        assert_eq!(w.candidate_knobs()["ring"], "3");
        assert_eq!(w.candidate_knobs()["variant"], "128x128x8-set-r3");
        let obj = w.candidate().unwrap();
        assert_eq!(obj.desc["rip"]["identical_to_generator"], true);
        assert_eq!(obj.launch_symbol, "gemm_mq6g256v2_mmq_i8_pm_set_gfx1201");
        assert!(new(
            shape,
            &mut knobs(&[("tile", "128x128"), ("ring", "2")]),
            Some(&path)
        )
        .is_ok());
        // `--ring 3` with a depth-2 script, or the 128x256 tile with the depth-3 one, is refused.
        let p2 = dir.join("ring2.rip");
        std::fs::write(&p2, TWIN).unwrap();
        let e = new(shape, &mut knobs(&[("ring", "3")]), Some(&p2))
            .err()
            .unwrap();
        assert!(e.contains("conflicts"), "{e}");
        let e = new(shape, &mut knobs(&[("tile", "128x256")]), Some(&path))
            .err()
            .unwrap();
        assert!(e.contains("ring depth 3 does not fit"), "{e}");
    }

    /// The twin with `const RING = 3` is identical to the generator's depth-3 kernel.
    #[test]
    fn rip_ring3_twin_is_identical_to_the_depth_3_generator() {
        let twin3 = TWIN.replace("const RING: u32 = 2;", "const RING: u32 = 3;");
        assert_ne!(twin3, TWIN);
        for (epi, raster) in [("set", "banded"), ("set_bf16", "linear")] {
            let spec = parse_spec(epi, raster, "128x128", "3").unwrap();
            let (_, symbol, info) = compile_rip(&twin3, spec).unwrap();
            assert_eq!(symbol, spec.symbol());
            assert_eq!(info["identical_to_generator"], true, "{epi} {raster}");
        }
        // The depth-2 script against a depth-3 spec is not the generator's program: its ABI
        // (the kernel's static LDS) differs, and the comparison refuses it.
        let spec = parse_spec("set", "banded", "128x128", "3").unwrap();
        let e = compile_rip(TWIN, spec).unwrap_err();
        assert!(e.contains("ABI mismatch"), "{e}");
    }
}
