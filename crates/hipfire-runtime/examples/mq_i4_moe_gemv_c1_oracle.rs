// SPDX-License-Identifier: Apache-2.0
//! K1-C1 / K1-Cn slab decode GEMV GPU oracle: byte-exactness (G1) and timing against the incumbent K1
//! path, swept over decode lane counts. Never participates in runtime dispatch. gfx1151 only.
//!
//! The incumbent is today's K1 route, driven exactly as dispatch drives it: `Gpu::qwen4_slab_union_plan`
//! once per layer step, then `Gpu::mq_i4_moe_gemv_slab_with(.., lanes, Some(SlabK1Module::K1))` for gate/up
//! and for down (union plan + per-codec-bucket one-wave-per-row GEMVs of the embedded
//! `mq_i4_moe_gemv_pm_gfx1151` image; K1 is FORCED, so `HIPFIRE_QWEN4_SLAB_K1_CN` plays no part).
//! The candidate depends on `lanes`:
//! * `lanes == 1`: K1-C1, `Gpu::mq_i4_moe_gemv_slab_c1` (one launch per projection, both codecs, no union
//!   plan; embedded `mq_i4_moe_gemv_c1_pm_gfx1151` image), called directly (independent of
//!   `HIPFIRE_QWEN4_SLAB_K1_C1`).
//! * `2 <= lanes <= 8`: K1-Cn, the SAME union plan (run once, shared by both) then
//!   `Gpu::mq_i4_moe_gemv_slab_with(.., lanes, Some(SlabK1Module::Cn))` (same per-codec bucket launches of the
//!   embedded `mq_i4_moe_gemv_cn_pm_gfx1151` image), also independent of the env switch.
//!
//! # G1 mode (default)
//! `--lanes LIST` (default `1,2,3,4,5,8`, values 1..=8, no duplicates). For every case, every lanes value and
//! every draw both paths run on the same inputs into sentinel-filled output buffers (word `0x7fc0dead`).
//! The buffers are allocated for the MAXIMUM lane count plus one slot row (gate/up Y0 and Y1:
//! `(80 + 1) * 640` words, down Y0: `(80 + 1) * 2560` words), so past the written extent of `lanes` lanes
//! (`10 * lanes` slot rows) there is always a sentinel tail of at least one slot row. Every F32 word of every
//! buffer, the sentinel tail included, is compared BITWISE between K1 and the candidate (signed zeros and NaN
//! payloads are differences). Independently of the comparison a path is also a failure if it leaves a
//! sentinel word inside the extent (unwritten) or disturbs a tail word.
//!
//! Inputs per draw: topk `[lanes * 10]` i32 (slot = lane * 10 + rank, 10 distinct ids per lane). Lane 0 is the
//! draw of the case (below); every later lane copies a subset of lane 0's ids (distinct within its lane, fresh
//! ids from the layer's mapped experts for the rest, then shuffled so ranks differ). Overlap by draw index
//! `d % 4`: 0 and 3 = 6 of 10 ids per lane (~60%, MTP-window-like expert overlap), 1 = all 10 ids in every lane
//! (a permutation: up to `lanes` slots per routed expert, the K1 `n <= 8` extreme), 2 = per lane `[0, 6, 3, 8]`
//! copied ids by `lane % 4`. X: gate `[lanes, 2560]` (lane row `style + lane`), down `[10 * lanes, 640]` (slot
//! row `style + slot`); the unused rest of the max-size X buffers is zero. X rows past `x_src_rows` are NOT
//! tested: the real dispatch cannot produce them (gate `x_src_rows = lanes` and every slot's row is
//! `slot / 10 < lanes`; down `x_src_rows = 10 * lanes` and the row is the slot itself), so no case exists.
//!
//! Cases (all through the product geometry: one arena per layer, experts `[gate_up][down]` back to
//! back at 256 B alignment, a 4096 B zero tail, a `[512]` u64 pointer table per projection and the
//! `[512]` u8 codec table with 0 = MQ4, 1 = W3), each repeated for every `--lanes` value
//! (case name suffix `/lanes<N>`):
//! * `syn_uniform_w3`, `syn_uniform_mq4`: a pool of 24 experts of one codec (ids always include 0,
//!   1, 510, 511); run as `SlabBuckets::Uniform` AND as a `SlabBuckets::Table` of all one codec.
//! * `syn_mixed_a`, `syn_mixed_b`, `syn_mixed_stress_x`: a pool of 24 experts with random codecs
//!   (>= 8 of each), `SlabBuckets::Table` (non-pool ids carry random codec bytes and a null
//!   pointer: a kernel that touches an unrouted expert faults or mismatches).
//! * Synthetic tensors are filled with RANDOM codes (random bytes in the W3 code plane / MQ4
//!   payload) and per-(row, group) scale classes: W3 f16 scales `0x0000`, `0x8000`, `+-0x0001`,
//!   `+-0x0400`, random subnormals, `1.0`, `+0x7bff` / `-0x7bff` (+-65504) and plain random
//!   normals of both signs; MQ4 symmetric headers `(sc, z = f16(-8 sc))` with `sc` over the same
//!   classes where the finite maximum is f16 `0x6fff` (`z = -65504`) / `0xefff`. Two of every
//!   three experts are "special" (every class), the third is plain random scales only.
//! * X rows rotate over six styles: normal-range, wide exponents, `+-2^24`, `+-0` / `+-1e-30`,
//!   F32 subnormals with `+-0`, and near-overflow magnitudes.
//! * Per case 12 lane-0 draws over the pool: draw 0 = ids {0, 511, ..} with 0 at rank 0 and 511 at
//!   rank 9, draw 1 its reverse, then an all-first-codec / all-second-codec draw, a 5/5 draw and
//!   random distinct draws.
//! * Real layers `real_L<l>` from `--artifact <sealed slab .mq3-xts> --layers L,L,..`: the layer's
//!   arena is read with the product's own plan (`expert_slabs::parse` ->
//!   `expert_slab_table::build_expert_slab_table` -> `expert_arena::ExpertArenaPlan::from_table`;
//!   `ExpertArenaLayer::{file_offset, bytes, experts[].{gate_up_offset, down_offset}}`,
//!   `codec_table()`), uploaded as ONE tensor plus `SLAB_ARENA_TAIL_PAD_BYTES` zeros exactly like the
//!   resident slab upload, with pointer tables `arena + offset` and the layer's codec table.
//!   Only the requested layers are read, one layer in host memory at a time. `--draws` (default
//!   64, minimum 64) lane-0 draws per layer per lanes value (+ the special draws above, counted in
//!   the total; `--hot-frac F` draws each slot from the MQ4 pool with probability F, default
//!   uniform over the 512 experts). A layer that has a single codec is also run `Uniform`.
//!
//! # Timing mode (`--timing`, needs `--artifact` + `--layers`)
//! Per lanes value: a draw pool of `--draws` (default 256, >= 64) routes (lane 0 = a distinct-top10 draw
//! assigned round-robin to the requested layers, later lanes derived as above; pass several layers so the
//! pool far exceeds the 32 MiB MALL; one draw already touches ~26 MiB of weights). First the first 16 draws
//! are G1-checked at that lanes value (refuses to time a wrong kernel; they appear as `timing_precheck/lanes<N>`
//! cases). Then, per rep, every sequence runs once as `--launches` (>= 64) back-to-back launches of
//! consecutive draws between two device events (one `hipEventElapsedTime` pair, the draw cursor advances
//! `--launches` per rep, all sequences of a rep use the same draws, sequence order rotates per rep); the
//! first `--warmup` reps are discarded. Sequences: `k1_plan` (union plan only), `k1_plan_gate` (plan +
//! gate/up buckets), `k1_plan_gate_down` (plan + gate/up + down); then at `lanes == 1` `c1_gate`, `c1_down`,
//! `c1_gate_down`, at `lanes >= 2` `cn_plan_gate` (the shared plan + Cn gate/up) and `cn_plan_gate_down`.
//! Median microseconds per launch of each sequence are reported; per projection (`gate_up`, `down`):
//! K1 gate/up = `k1_plan_gate`, K1 down = `k1_plan_gate_down - k1_plan_gate` (derived); `lanes == 1`: C1 gate/up =
//! `c1_gate`, C1 down = `c1_down`; `lanes >= 2`: Cn gate/up = `cn_plan_gate`, Cn down = `cn_plan_gate_down -
//! cn_plan_gate` (the union plan is shared and counted in both gate/up figures, as K1 always counted it).
//! GB/s = mean unique weight bytes of the routed experts of the step (experts shared between lanes counted
//! once) of that projection / median time.
//!
//! # CLI
//! ```text
//! mq_i4_moe_gemv_c1_oracle [--artifact ABS.mq3-xts --layers 0,1,23,47] [--draws N] [--hot-frac F]
//!     [--lanes 1,2,3,4,5,8] [--seed N] [--skip-synthetic] [--only SUBSTR]...
//!     [--timing --reps 30 --warmup 5 --launches 64]
//!     [--device N] [--expect-arch gfx1151] [--expect-uuid GPU-..] [--expect-bdf 0000:..:..]
//!     [--card NAME] [--fabric-owner OWNER] [--fabric-wrapper PATH] [--fabric-card NAME]
//! ```
//! `--expect-arch` defaults to `gfx1151`; `--expect-uuid` / `--expect-bdf` are enforced when given.
//! `--card/--fabric-*` (env `FABRIC_CARD/FABRIC_OWNER/FABRIC_WRAPPER`) are recorded verbatim.
//!
//! # Output
//! ONE JSON line on stdout, schema `mq-i4-moe-gemv-c1-oracle-v2`:
//! `{schema, mode: "g1"|"timing", command: {argv, build_command}, device: {index, uuid, bdf, arch,
//! expected_*, identity_match}, modules: {k1: {module, symbols, sha256, bytes}, c1: {..}, cn: {..}},
//! config: {.., lanes: [..]}, artifact?, cases: [{name, origin, layer?, buckets, lanes, candidate: "c1"|"cn",
//! draws, words_compared, mismatch_count, first_mismatches: [{draw, part, word_index, slot, lane, row,
//! k1_bits, cand_bits, topk}], k1_unwritten_words, cand_unwritten_words, k1_tail_disturbed_words,
//! cand_tail_disturbed_words, draws_with_both_codecs, pass}], case_counts, mismatch_count, byte_exact, pass,
//! rc, timing?: {event_timing, layers, draws, reps, warmup, launches_per_sample, mall_bytes, fabric,
//! timing_source, verified_byte_exact_before_timing, by_lanes: [{lanes, candidate, draws_per_sample,
//! pool_unique_weight_bytes, min_sample_unique_weight_bytes, sample_working_set_exceeds_mall,
//! median_us: {<sequence>: us}, projections: {gate_up: {unique_bytes_mean, k1_us, cand_us, k1_gbps,
//! cand_gbps, cand_speedup}, down: {..}, k1_down_is, cand_down_is}, gate_down_total: {k1_us, cand_us,
//! cand_speedup, unique_bytes_mean, k1_gbps, cand_gbps}, samples_us: {<sequence>: [us]}}]}}`. Exit code 1 on
//! any mismatch or error (the error text is then also on stderr and in `error`).
//!
//! # Build / run
//! ```text
//! cargo build -p hipfire-runtime --release --features lab --example mq_i4_moe_gemv_c1_oracle
//! # G1 (synthetic + real layers, lanes 1,2,3,4,5,8; add --lanes 1,2,3,4,5,6,7,8 for the full sweep):
//! target/release/examples/mq_i4_moe_gemv_c1_oracle --artifact /abs/qwen3.8-flash-next-base.mq3-xts \
//!   --layers 0,1,23,47 --draws 64 --expect-arch gfx1151 --expect-uuid GPU-<uuid> --expect-bdf 0000:<bdf>
//! # Timing (separate run, after G1 passed):
//! target/release/examples/mq_i4_moe_gemv_c1_oracle --timing --artifact /abs/qwen3.8-flash-next-base.mq3-xts \
//!   --layers 0,1,23,47 --draws 256 --reps 30 --warmup 5 --launches 64 \
//!   --expect-arch gfx1151 --expect-uuid GPU-<uuid> --expect-bdf 0000:<bdf>
//! # Local CPU known-answer tests (no GPU):
//! cargo test -p hipfire-runtime --release --features lab --example mq_i4_moe_gemv_c1_oracle
//! ```
use hipfire_arch_qwen4::expert_arena::{ExpertArenaPlan, CODEC_MQ4, CODEC_W3};
use hipfire_arch_qwen4::expert_slab_table::build_expert_slab_table;
use hipfire_arch_qwen4::expert_slabs::{self, ExpertFormat};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::weight_store::SLAB_ARENA_TAIL_PAD_BYTES;
use rdna_compute::slab_moe::{
    SlabBuckets, SlabK1Module, MQ_I4_MOE_GEMV_C1_SYMBOLS, MQ_I4_MOE_GEMV_CN_SYMBOLS,
    MQ_I4_MOE_GEMV_SYMBOLS, SLAB_DOWN_K, SLAB_DOWN_M, SLAB_EXPERTS, SLAB_GATE_UP_K, SLAB_GATE_UP_M,
    SLAB_NARROW_MAX_LANES, SLAB_TOP_K,
};
use rdna_compute::{kernels, DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::os::unix::fs::FileExt;
use std::path::Path;

type Result<T, E = String> = std::result::Result<T, E>;
fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

const SCHEMA: &str = "mq-i4-moe-gemv-c1-oracle-v2";
const BUILD_COMMAND: &str =
    "cargo build -p hipfire-runtime --release --features lab --example mq_i4_moe_gemv_c1_oracle";
const K1_MODULE: &str = "mq_i4_moe_gemv_pm_gfx1151";
const C1_MODULE: &str = "mq_i4_moe_gemv_c1_pm_gfx1151";
const CN_MODULE: &str = "mq_i4_moe_gemv_cn_pm_gfx1151";
const DEFAULT_ARCH: &str = "gfx1151";
const DEFAULT_LANES: [usize; 6] = [1, 2, 3, 4, 5, 8];
/// Unwritten output words (a quiet NaN with a payload no kernel produces).
const SENTINEL_BITS: u32 = 0x7fc0_dead;
const MALL_BYTES: u64 = 32 << 20;
const GATE_Y_ROW: usize = SLAB_GATE_UP_M / 2;
const DOWN_Y_ROW: usize = SLAB_DOWN_M;
const POOL: usize = 24;
const SYNTH_DRAWS: usize = 12;
const MIN_DRAWS: usize = 64;
const TIMING_VERIFY_DRAWS: usize = 16;
const ARENA_ALIGN: usize = 256;
const UNMAPPED: usize = usize::MAX;

// ---------------------------------------------------------------------------------------
// Small utilities
// ---------------------------------------------------------------------------------------

struct Rng(u64);
fn mix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(mix64(seed))
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        mix64(self.0)
    }
    fn u32(&mut self) -> u32 {
        (self.next() >> 32) as u32
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = self.below(i + 1);
            v.swap(i, j);
        }
    }
    fn fill(&mut self, out: &mut [u8]) {
        for chunk in out.chunks_mut(8) {
            let word = self.next().to_le_bytes();
            chunk.copy_from_slice(&word[..chunk.len()]);
        }
    }
}
fn name_seed(name: &str) -> u64 {
    u64::from_le_bytes(Sha256::digest(name.as_bytes())[..8].try_into().unwrap())
}
fn hex(b: &[u8]) -> String {
    b.iter().map(|v| format!("{v:02x}")).collect()
}
fn f16f(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}
/// Symmetric MQ4 zero `f16(-8 * sc)`.
fn sym_zero_bits(sc: u16) -> u16 {
    half::f16::from_f32(-8.0 * f16f(sc)).to_bits()
}
fn bits_hex(v: u32) -> String {
    format!("0x{v:08x}")
}
fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn u64_bytes(v: &[u64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    }
}
fn sha_of(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Codec {
    Mq4,
    W3,
}
impl Codec {
    /// Per-layer codec table byte: 0 = MQ4, 1 = W3.
    fn byte(self) -> u8 {
        match self {
            Codec::Mq4 => CODEC_MQ4,
            Codec::W3 => CODEC_W3,
        }
    }
    fn from_byte(b: u8) -> Codec {
        if b == CODEC_W3 {
            Codec::W3
        } else {
            Codec::Mq4
        }
    }
    fn format(self) -> ExpertFormat {
        match self {
            Codec::Mq4 => ExpertFormat::Mq4Xts,
            Codec::W3 => ExpertFormat::Mq3I4,
        }
    }
    /// K1 bucket index (`SlabBuckets::Uniform`): 0 = W3, 1 = MQ4.
    fn k1_bucket(self) -> usize {
        match self {
            Codec::W3 => 0,
            Codec::Mq4 => 1,
        }
    }
}

// ---------------------------------------------------------------------------------------
// Synthetic tensors and activations
// ---------------------------------------------------------------------------------------

/// f16 scale bits of scale class `class` (0..12). `max` is the finite maximum of the codec
/// (W3 `0x7bff`, MQ4 `0x6fff`, whose `-8 sc` is still finite).
fn scale_of_class(class: usize, max: u16, rng: &mut Rng) -> u16 {
    match class % 12 {
        0 => max,
        1 => max | 0x8000,
        2 => 0x0000,
        3 => 0x8000,
        4 => 0x0001,
        5 => 0x8001,
        6 => 1 + rng.below(0x3ff) as u16,
        7 => 0x0400,
        8 => 0x8400,
        9 => 0x3c00,
        _ => plain_scale(rng),
    }
}
/// A random normal scale of either sign (exponent field 8..=20, i.e. 2^-7 .. 2^5).
fn plain_scale(rng: &mut Rng) -> u16 {
    let sign = (rng.u32() as u16 & 1) << 15;
    sign | (((8 + rng.below(13)) as u16) << 10) | (rng.u32() as u16 & 0x3ff)
}

/// One expert projection tensor of `codec` with random codes and the scale classes above,
/// laid out exactly as the slab payload (`ExpertFormat::{gate_up,down}_bytes`).
fn gen_tensor(codec: Codec, down: bool, salt: usize, special: bool, rng: &mut Rng) -> Vec<u8> {
    let (m, k) = if down {
        (SLAB_DOWN_M, SLAB_DOWN_K)
    } else {
        (SLAB_GATE_UP_M, SLAB_GATE_UP_K)
    };
    let class = |r: usize, g: usize, rng: &mut Rng, max: u16| -> u16 {
        if special {
            scale_of_class((r * 5 + g * 3 + salt) % 12, max, rng)
        } else {
            plain_scale(rng)
        }
    };
    match codec {
        Codec::W3 => {
            let plane = m * k * 3 / 8;
            let groups = k / 128;
            let mut out = vec![0u8; plane + m * groups * 2];
            rng.fill(&mut out[..plane]);
            for r in 0..m {
                for g in 0..groups {
                    let sc = class(r, g, rng, 0x7bff);
                    let at = plane + 2 * (r * groups + g);
                    out[at..at + 2].copy_from_slice(&sc.to_le_bytes());
                }
            }
            out
        }
        Codec::Mq4 => {
            // gate: 10 groups of 136 B per row (two 4 B headers, 128 B payload);
            // down: 5 groups of 68 B (one 4 B header, 64 B payload).
            let (stride, headers, payload) = if down { (68, 1, 64) } else { (136, 2, 128) };
            let groups = k / (if down { 128 } else { 256 });
            let mut out = vec![0u8; m * groups * stride];
            rng.fill(&mut out);
            for r in 0..m {
                for g in 0..groups {
                    let base = (r * groups + g) * stride;
                    for h in 0..headers {
                        let sc = class(r, g * headers + h, rng, 0x6fff);
                        out[base + 4 * h..base + 4 * h + 2].copy_from_slice(&sc.to_le_bytes());
                        out[base + 4 * h + 2..base + 4 * h + 4]
                            .copy_from_slice(&sym_zero_bits(sc).to_le_bytes());
                    }
                    debug_assert_eq!(headers * 4 + payload, stride);
                }
            }
            out
        }
    }
}

/// X row of `n` F32 in one of six styles.
fn gen_x_row(style: usize, n: usize, rng: &mut Rng) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let v = match style % 6 {
            0 => {
                let (s, e, m) = (
                    rng.u32() >> 31,
                    124 + rng.below(5) as u32,
                    rng.u32() & 0x7f_ffff,
                );
                f32::from_bits((s << 31) | (e << 23) | m)
            }
            1 => {
                let (s, e, m) = (
                    rng.u32() >> 31,
                    107 + rng.below(24) as u32,
                    rng.u32() & 0x7f_ffff,
                );
                f32::from_bits((s << 31) | (e << 23) | m)
            }
            2 => [16777216.0f32, 1.0, -16777216.0, 1.0][i % 4],
            3 => [0.0f32, -0.0, 1.0e-30, -1.0e-30, 1.0, -1.0, 0.5, -0.5][i % 8],
            4 => {
                // F32 subnormals (exponent field 0) with signed zeros sprinkled in.
                if i % 7 == 0 {
                    if i % 14 == 0 {
                        0.0
                    } else {
                        -0.0
                    }
                } else {
                    f32::from_bits(((rng.u32() >> 31) << 31) | (rng.u32() & 0x7f_ffff))
                }
            }
            _ => {
                let (s, e, m) = (
                    rng.u32() >> 31,
                    246 + rng.below(8) as u32,
                    rng.u32() & 0x7f_ffff,
                );
                f32::from_bits((s << 31) | (e << 23) | m)
            }
        };
        out.push(v);
    }
    out
}

struct Xs {
    gate: Vec<f32>,
    down: Vec<f32>,
}
/// X of `lanes` decode lanes: gate `[lanes, 2560]` (lane row `style + lane`), down `[10 * lanes, 640]`
/// (slot row `style + slot`). At `lanes == 1` this is the one-lane X of the K1-C1 oracle.
fn gen_xs(style: usize, lanes: usize, rng: &mut Rng) -> Xs {
    let mut gate = Vec::with_capacity(lanes * SLAB_GATE_UP_K);
    for lane in 0..lanes {
        gate.extend(gen_x_row(style + lane, SLAB_GATE_UP_K, rng));
    }
    let mut down = Vec::with_capacity(SLAB_TOP_K * lanes * SLAB_DOWN_K);
    for u in 0..SLAB_TOP_K * lanes {
        down.extend(gen_x_row(style + u, SLAB_DOWN_K, rng));
    }
    Xs { gate, down }
}

// ---------------------------------------------------------------------------------------
// Layers (host arena -> device arena + tables)
// ---------------------------------------------------------------------------------------

/// A layer's arena bytes (including the zero tail) and per-expert placement, on the host.
struct HostLayer {
    arena: Vec<u8>,
    gate_off: Vec<usize>,
    down_off: Vec<usize>,
    /// `[512]` per-expert codec byte (0 = MQ4, 1 = W3).
    codec: Vec<u8>,
}

/// A layer resident on the device with its product tables.
struct Layer {
    name: String,
    arena: GpuTensor,
    gate_ptrs: GpuTensor,
    down_ptrs: GpuTensor,
    codec_table: GpuTensor,
    codec: Vec<u8>,
    mapped: Vec<bool>,
}
impl Layer {
    fn ids_of(&self, codec: Option<Codec>) -> Vec<u32> {
        (0..SLAB_EXPERTS as u32)
            .filter(|&e| {
                self.mapped[e as usize]
                    && codec.is_none_or(|c| Codec::from_byte(self.codec[e as usize]) == c)
            })
            .collect()
    }
    fn codec_of(&self, id: i32) -> Codec {
        Codec::from_byte(self.codec[id as usize])
    }
    fn gate_bytes(&self, id: i32) -> usize {
        self.codec_of(id).format().gate_up_bytes()
    }
    fn down_bytes(&self, id: i32) -> usize {
        self.codec_of(id).format().down_bytes()
    }
}

fn up_raw(gpu: &mut Gpu, bytes: &[u8]) -> Result<GpuTensor> {
    let t = gpu
        .alloc_tensor(&[bytes.len().max(4)], DType::Raw)
        .map_err(err)?;
    gpu.hip.memcpy_htod(&t.buf, bytes).map_err(err)?;
    Ok(t)
}
fn alloc_f32(gpu: &mut Gpu, words: usize) -> Result<GpuTensor> {
    gpu.alloc_tensor(&[words], DType::F32).map_err(err)
}

fn upload_layer(gpu: &mut Gpu, name: &str, h: HostLayer) -> Result<Layer> {
    let arena = up_raw(gpu, &h.arena)?;
    let base = arena.buf.as_ptr() as usize as u64;
    let table = |offs: &[usize]| -> Vec<u8> {
        u64_bytes(
            &offs
                .iter()
                .map(|&o| if o == UNMAPPED { 0 } else { base + o as u64 })
                .collect::<Vec<u64>>(),
        )
    };
    let gate_ptrs = up_raw(gpu, &table(&h.gate_off))?;
    let down_ptrs = up_raw(gpu, &table(&h.down_off))?;
    let codec_table = up_raw(gpu, &h.codec)?;
    let mapped = h.gate_off.iter().map(|&o| o != UNMAPPED).collect();
    Ok(Layer {
        name: name.to_string(),
        arena,
        gate_ptrs,
        down_ptrs,
        codec_table,
        codec: h.codec,
        mapped,
    })
}

fn free_layer(gpu: &mut Gpu, l: Layer) -> Result<()> {
    for t in [l.arena, l.gate_ptrs, l.down_ptrs, l.codec_table] {
        gpu.free_tensor(t).map_err(err)?;
    }
    Ok(())
}

fn align_up(v: usize) -> usize {
    v.div_ceil(ARENA_ALIGN) * ARENA_ALIGN
}

/// Synthetic layer: `pool` ids (with their codecs) laid out as the product arena; every other
/// id is unmapped (null pointer) and carries `fill_codec` table bytes.
fn synth_layer(
    name: &str,
    pool: &[(u32, Codec)],
    fill: &mut dyn FnMut(&mut Rng) -> Codec,
    rng: &mut Rng,
) -> HostLayer {
    let mut arena = Vec::new();
    let mut gate_off = vec![UNMAPPED; SLAB_EXPERTS];
    let mut down_off = vec![UNMAPPED; SLAB_EXPERTS];
    let mut codec: Vec<u8> = (0..SLAB_EXPERTS).map(|_| fill(rng).byte()).collect();
    for (i, &(id, c)) in pool.iter().enumerate() {
        let special = (name_seed(name) as usize + i) % 3 != 2;
        let gate = gen_tensor(c, false, i + id as usize, special, rng);
        let down = gen_tensor(c, true, i + id as usize, special, rng);
        debug_assert_eq!(gate.len(), c.format().gate_up_bytes());
        debug_assert_eq!(down.len(), c.format().down_bytes());
        arena.resize(align_up(arena.len()), 0);
        gate_off[id as usize] = arena.len();
        arena.extend_from_slice(&gate);
        down_off[id as usize] = arena.len();
        arena.extend_from_slice(&down);
        codec[id as usize] = c.byte();
    }
    arena.resize(arena.len() + SLAB_ARENA_TAIL_PAD_BYTES, 0);
    HostLayer {
        arena,
        gate_off,
        down_off,
        codec,
    }
}

/// Read one trunk layer's arena of the sealed slab artifact exactly as the product loader plans it.
struct Artifact {
    path: String,
    hfq: HfqFile,
    plan: ExpertArenaPlan,
}

fn open_artifact(path: &str) -> Result<Artifact> {
    let hfq = HfqFile::open(Path::new(path)).map_err(|e| format!("{path}: {e}"))?;
    let meta = expert_slabs::parse(&hfq.metadata_json)
        .map_err(|e| format!("{path}: qwen4.expert_slabs metadata: {e:?}"))?
        .ok_or_else(|| {
            format!("{path}: not an expert-slab artifact (no qwen4.expert_slabs metadata)")
        })?;
    let table = build_expert_slab_table(&meta, hfq.tensors())
        .map_err(|e| format!("{path}: slab table: {e:?}"))?;
    let counts = table.counts();
    if counts.w2 != 0 {
        return Err(format!(
            "{path}: {} W2 experts; the slab route has no W2 kernels",
            counts.w2
        ));
    }
    let plan = ExpertArenaPlan::from_table(&table).map_err(|e| format!("{path}: arenas: {e}"))?;
    plan.check_executable(DEFAULT_ARCH)
        .map_err(|e| format!("{path}: arenas: {e}"))?;
    Ok(Artifact {
        path: path.to_string(),
        hfq,
        plan,
    })
}

fn load_real_layer(art: &Artifact, layer: usize) -> Result<(HostLayer, Value)> {
    let a = art
        .plan
        .layer(layer)
        .ok_or_else(|| format!("layer {layer} is outside the artifact's trunk"))?;
    let codec = a.codec_table().map_err(|e| format!("layer {layer}: {e}"))?;
    let (mq4, w3, e8p) = a.codec_counts();
    if e8p != 0 {
        return Err(format!(
            "layer {layer}: {e8p} E8P experts; K1-C1 / K1-Cn serve MQ4 + W3 layers only"
        ));
    }
    let mut arena = vec![0u8; a.bytes + SLAB_ARENA_TAIL_PAD_BYTES];
    let file = std::fs::File::open(&art.path).map_err(|e| format!("{}: {e}", art.path))?;
    file.read_exact_at(&mut arena[..a.bytes], a.file_offset as u64)
        .map_err(|e| {
            format!(
                "layer {layer}: reading {} arena bytes at {}: {e}",
                a.bytes, a.file_offset
            )
        })?;
    let gate_off = a.experts.iter().map(|e| e.gate_up_offset).collect();
    let down_off = a.experts.iter().map(|e| e.down_offset).collect();
    let meta = json!({"layer": layer, "file_offset": a.file_offset, "arena_bytes": a.bytes,
        "tail_pad_bytes": SLAB_ARENA_TAIL_PAD_BYTES, "experts": {"mq4": mq4, "w3": w3}});
    Ok((
        HostLayer {
            arena,
            gate_off,
            down_off,
            codec,
        },
        meta,
    ))
}

// ---------------------------------------------------------------------------------------
// Draws
// ---------------------------------------------------------------------------------------

fn pick_distinct(rng: &mut Rng, pool: &[u32], n: usize) -> Vec<u32> {
    let mut v = pool.to_vec();
    rng.shuffle(&mut v);
    v.truncate(n);
    v
}

/// `n` routes (10 distinct ids, slot = rank) over the layer's mapped experts. With `specials` the
/// first draws are the edge draws (0 at rank 0 and 511 at rank 9, then reversed, then all-hot,
/// all-cold and 5/5 where the codec pools allow). `hot_frac` draws each slot from the MQ4 pool
/// with that probability (distinct), else the draw is uniform over the mapped experts.
fn gen_draws(
    rng: &mut Rng,
    layer: &Layer,
    n: usize,
    hot_frac: Option<f64>,
    specials: bool,
) -> Vec<Vec<i32>> {
    let all = layer.ids_of(None);
    let hot = layer.ids_of(Some(Codec::Mq4));
    let cold = layer.ids_of(Some(Codec::W3));
    let mut draws: Vec<Vec<i32>> = Vec::new();
    let ids = |v: Vec<u32>| v.into_iter().map(|e| e as i32).collect::<Vec<i32>>();
    if specials {
        if all.contains(&0) && all.contains(&(SLAB_EXPERTS as u32 - 1)) && all.len() >= SLAB_TOP_K {
            let rest: Vec<u32> = all
                .iter()
                .copied()
                .filter(|&e| e != 0 && e != SLAB_EXPERTS as u32 - 1)
                .collect();
            let mut middle = pick_distinct(rng, &rest, SLAB_TOP_K - 2);
            middle.insert(0, 0);
            middle.push(SLAB_EXPERTS as u32 - 1);
            let d = ids(middle);
            let mut rev = d.clone();
            rev.reverse();
            draws.push(d);
            draws.push(rev);
        }
        if hot.len() >= SLAB_TOP_K {
            draws.push(ids(pick_distinct(rng, &hot, SLAB_TOP_K)));
        }
        if cold.len() >= SLAB_TOP_K {
            draws.push(ids(pick_distinct(rng, &cold, SLAB_TOP_K)));
        }
        if hot.len() >= 5 && cold.len() >= 5 {
            let mut v = pick_distinct(rng, &hot, 5);
            v.extend(pick_distinct(rng, &cold, 5));
            rng.shuffle(&mut v);
            draws.push(ids(v));
        }
        draws.truncate(n);
    }
    while draws.len() < n {
        let route = match hot_frac {
            Some(f) if !hot.is_empty() && !cold.is_empty() => {
                let want_hot = (0..SLAB_TOP_K).filter(|_| rng.unit() < f).count();
                let n_hot = want_hot
                    .min(hot.len())
                    .max(SLAB_TOP_K.saturating_sub(cold.len()));
                let mut v = pick_distinct(rng, &hot, n_hot);
                v.extend(pick_distinct(rng, &cold, SLAB_TOP_K - n_hot));
                rng.shuffle(&mut v);
                v
            }
            _ => pick_distinct(rng, &all, SLAB_TOP_K),
        };
        draws.push(ids(route));
    }
    draws
}

fn style_for(real: bool, case_idx: usize, draw: usize) -> usize {
    if real {
        if draw % 4 == 3 {
            2 + (draw / 4) % 4
        } else {
            draw % 2
        }
    } else {
        (draw + case_idx) % 6
    }
}

const LANE_KEEP_SPREAD: [usize; 4] = [0, 6, 3, 8];

/// Ids a later lane copies from lane 0 at draw index `draw`: `d % 4` 0 and 3 -> 6 of 10 (~60%),
/// 1 -> all 10 (a permutation), 2 -> `[0, 6, 3, 8][lane % 4]`.
fn lane_keep(draw: usize, lane: usize) -> usize {
    match draw % 4 {
        1 => SLAB_TOP_K,
        2 => LANE_KEEP_SPREAD[lane % 4],
        _ => 6,
    }
}

/// Flat topk `[lanes * 10]` (slot = lane * 10 + rank): lane 0 is `base`; every later lane keeps
/// `lane_keep` random ids of `base`, fills up with distinct fresh ids of `all` (the mapped experts)
/// and is shuffled, so every lane holds 10 distinct ids at its own ranks while experts repeat
/// across lanes like an MTP window. `lanes == 1` returns `base`.
fn lane_routes(rng: &mut Rng, all: &[u32], base: &[i32], lanes: usize, draw: usize) -> Vec<i32> {
    let mut out = base.to_vec();
    for lane in 1..lanes {
        let keep = lane_keep(draw, lane);
        let mut route = base.to_vec();
        rng.shuffle(&mut route);
        route.truncate(keep);
        let fresh: Vec<u32> = all
            .iter()
            .copied()
            .filter(|e| !route.contains(&(*e as i32)))
            .collect();
        route.extend(
            pick_distinct(rng, &fresh, SLAB_TOP_K - keep)
                .into_iter()
                .map(|e| e as i32),
        );
        rng.shuffle(&mut route);
        out.extend(route);
    }
    out
}

// ---------------------------------------------------------------------------------------
// Device IO and the two paths
// ---------------------------------------------------------------------------------------

/// The candidate under test: K1-C1 at one lane, K1-Cn at 2..=8.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cand {
    C1,
    Cn,
}
impl Cand {
    fn for_lanes(lanes: usize) -> Cand {
        if lanes == 1 {
            Cand::C1
        } else {
            Cand::Cn
        }
    }
    fn name(self) -> &'static str {
        match self {
            Cand::C1 => "c1",
            Cand::Cn => "cn",
        }
    }
}

struct Outs {
    g0: GpuTensor,
    g1: GpuTensor,
    d0: GpuTensor,
}
struct Io {
    topk: GpuTensor,
    xg: GpuTensor,
    xd: GpuTensor,
    k1: Outs,
    cand: Outs,
}
const MAX_SLOTS: usize = SLAB_TOP_K * SLAB_NARROW_MAX_LANES;
/// Output allocations: the maximum-lane extent plus one slot row of sentinel tail.
const GATE_Y_ALLOC: usize = (MAX_SLOTS + 1) * GATE_Y_ROW;
const DOWN_Y_ALLOC: usize = (MAX_SLOTS + 1) * DOWN_Y_ROW;
const GATE_X_ALLOC: usize = SLAB_NARROW_MAX_LANES * SLAB_GATE_UP_K;
const DOWN_X_ALLOC: usize = MAX_SLOTS * SLAB_DOWN_K;

fn alloc_outs(gpu: &mut Gpu) -> Result<Outs> {
    Ok(Outs {
        g0: alloc_f32(gpu, GATE_Y_ALLOC)?,
        g1: alloc_f32(gpu, GATE_Y_ALLOC)?,
        d0: alloc_f32(gpu, DOWN_Y_ALLOC)?,
    })
}
fn alloc_io(gpu: &mut Gpu) -> Result<Io> {
    Ok(Io {
        topk: gpu
            .alloc_tensor(&[MAX_SLOTS * 4], DType::Raw)
            .map_err(err)?,
        xg: alloc_f32(gpu, GATE_X_ALLOC)?,
        xd: alloc_f32(gpu, DOWN_X_ALLOC)?,
        k1: alloc_outs(gpu)?,
        cand: alloc_outs(gpu)?,
    })
}
fn free_io(gpu: &mut Gpu, io: Io) -> Result<()> {
    for t in [
        io.topk, io.xg, io.xd, io.k1.g0, io.k1.g1, io.k1.d0, io.cand.g0, io.cand.g1, io.cand.d0,
    ] {
        gpu.free_tensor(t).map_err(err)?;
    }
    Ok(())
}
fn topk_bytes(topk: &[i32]) -> Vec<u8> {
    topk.iter().flat_map(|e| e.to_le_bytes()).collect()
}
/// F32 bytes of `v` zero-padded to `words` words.
fn padded_f32_bytes(v: &[f32], words: usize) -> Vec<u8> {
    assert!(v.len() <= words);
    let mut b = f32_bytes(v);
    b.resize(words * 4, 0);
    b
}
/// Upload `xs` into the max-size X buffers (the rows past the lanes in use are zero).
fn upload_xs(gpu: &Gpu, io: &Io, xs: &Xs) -> Result<()> {
    gpu.hip
        .memcpy_htod(&io.xg.buf, &padded_f32_bytes(&xs.gate, GATE_X_ALLOC))
        .map_err(err)?;
    gpu.hip
        .memcpy_htod(&io.xd.buf, &padded_f32_bytes(&xs.down, DOWN_X_ALLOC))
        .map_err(err)
}
fn sentinel_fill(gpu: &Gpu, t: &GpuTensor, words: usize) -> Result<()> {
    let host = vec![SENTINEL_BITS; words];
    let bytes: Vec<u8> = host.iter().flat_map(|w| w.to_le_bytes()).collect();
    gpu.hip.memcpy_htod(&t.buf, &bytes).map_err(err)
}
fn download_words(gpu: &Gpu, t: &GpuTensor, words: usize) -> Result<Vec<u32>> {
    let mut bytes = vec![0u8; words * 4];
    gpu.hip.memcpy_dtoh(&mut bytes, &t.buf).map_err(err)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

/// Cold codec of every oracle layer: W3 (`1`, K1 bucket 0); the other bucket is MQ4.
const COLD_W3: u8 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    /// `SlabBuckets::Uniform(K1 bucket)`; 0 = W3, 1 = MQ4.
    Uniform(usize),
    Table,
}
impl Mode {
    fn label(self) -> &'static str {
        match self {
            Mode::Uniform(0) => "uniform_w3",
            Mode::Uniform(_) => "uniform_mq4",
            Mode::Table => "table",
        }
    }
    fn buckets(self, l: &Layer) -> SlabBuckets<'_> {
        match self {
            Mode::Uniform(b) => SlabBuckets::Uniform(b),
            Mode::Table => SlabBuckets::Table(&l.codec_table),
        }
    }
}

fn k1_plan(gpu: &mut Gpu, l: &Layer, mode: Mode, topk: &GpuTensor, lanes: usize) -> Result<()> {
    gpu.qwen4_slab_union_plan(topk, mode.buckets(l), lanes)
        .map_err(err)
}
/// Gate/up through one forced K1-family module over the plan already run for `lanes`.
fn slab_gate(
    gpu: &mut Gpu,
    l: &Layer,
    mode: Mode,
    xg: &GpuTensor,
    o: &Outs,
    lanes: usize,
    m: SlabK1Module,
) -> Result<()> {
    gpu.mq_i4_moe_gemv_slab_with(
        false,
        mode.buckets(l),
        &l.gate_ptrs,
        xg,
        &o.g0,
        Some(&o.g1),
        lanes,
        COLD_W3,
        Some(m),
    )
    .map_err(err)
}
/// Down through one forced K1-family module over the plan already run for `lanes`.
fn slab_down(
    gpu: &mut Gpu,
    l: &Layer,
    mode: Mode,
    xd: &GpuTensor,
    o: &Outs,
    lanes: usize,
    m: SlabK1Module,
) -> Result<()> {
    gpu.mq_i4_moe_gemv_slab_with(
        true,
        mode.buckets(l),
        &l.down_ptrs,
        xd,
        &o.d0,
        None,
        lanes,
        COLD_W3,
        Some(m),
    )
    .map_err(err)
}
fn c1_gate(
    gpu: &mut Gpu,
    l: &Layer,
    mode: Mode,
    topk: &GpuTensor,
    xg: &GpuTensor,
    o: &Outs,
) -> Result<()> {
    gpu.mq_i4_moe_gemv_slab_c1(
        false,
        mode.buckets(l),
        &l.gate_ptrs,
        topk,
        xg,
        &o.g0,
        Some(&o.g1),
        COLD_W3,
    )
    .map_err(err)
}
fn c1_down(
    gpu: &mut Gpu,
    l: &Layer,
    mode: Mode,
    topk: &GpuTensor,
    xd: &GpuTensor,
    o: &Outs,
) -> Result<()> {
    gpu.mq_i4_moe_gemv_slab_c1(
        true,
        mode.buckets(l),
        &l.down_ptrs,
        topk,
        xd,
        &o.d0,
        None,
        COLD_W3,
    )
    .map_err(err)
}

// ---------------------------------------------------------------------------------------
// Bitwise comparison
// ---------------------------------------------------------------------------------------

#[derive(Default)]
struct Diff {
    words: u64,
    mismatches: u64,
    first: Vec<Value>,
    k1_unwritten: u64,
    cand_unwritten: u64,
    k1_tail_bad: u64,
    cand_tail_bad: u64,
}
impl Diff {
    fn clean(&self) -> bool {
        self.mismatches == 0
            && self.k1_unwritten == 0
            && self.cand_unwritten == 0
            && self.k1_tail_bad == 0
            && self.cand_tail_bad == 0
    }
    fn merge(&mut self, o: Diff) {
        self.words += o.words;
        self.mismatches += o.mismatches;
        for v in o.first {
            if self.first.len() < 8 {
                self.first.push(v);
            }
        }
        self.k1_unwritten += o.k1_unwritten;
        self.cand_unwritten += o.cand_unwritten;
        self.k1_tail_bad += o.k1_tail_bad;
        self.cand_tail_bad += o.cand_tail_bad;
    }
}

/// Compare two oversize outputs; the first `extent` words are the written region (`row` words per slot),
/// everything after it must still be the sentinel.
fn compare_part(
    part: &str,
    k1: &[u32],
    cand: &[u32],
    extent: usize,
    row: usize,
    draw: usize,
    topk: &[i32],
) -> Diff {
    let mut d = Diff {
        words: k1.len().max(cand.len()) as u64,
        ..Diff::default()
    };
    assert_eq!(k1.len(), cand.len());
    for (i, (&a, &b)) in k1.iter().zip(cand).enumerate() {
        if a != b {
            d.mismatches += 1;
            if d.first.len() < 8 {
                d.first.push(
                    json!({"draw": draw, "part": part, "word_index": i, "slot": i / row,
                    "lane": i / row / SLAB_TOP_K, "row": i % row,
                    "k1_bits": bits_hex(a), "cand_bits": bits_hex(b), "topk": topk}),
                );
            }
        }
        if i < extent {
            d.k1_unwritten += (a == SENTINEL_BITS) as u64;
            d.cand_unwritten += (b == SENTINEL_BITS) as u64;
        } else {
            d.k1_tail_bad += (a != SENTINEL_BITS) as u64;
            d.cand_tail_bad += (b != SENTINEL_BITS) as u64;
        }
    }
    d
}

/// One draw (`topk` = `[lanes * 10]` routed ids) through K1 and the candidate of `lanes` on `io`, compared
/// bitwise. The union plan runs ONCE; K1 gate/up and down are forced K1, the candidate is C1 (`lanes == 1`,
/// no plan) or forced Cn (`lanes >= 2`) over the same plan.
#[allow(clippy::too_many_arguments)]
fn g1_draw(
    gpu: &mut Gpu,
    l: &Layer,
    mode: Mode,
    io: &Io,
    topk: &[i32],
    lanes: usize,
    xs: &Xs,
    draw: usize,
) -> Result<Diff> {
    if topk.len() != lanes * SLAB_TOP_K {
        return Err(format!(
            "g1_draw: topk holds {} ids for {lanes} lanes",
            topk.len()
        ));
    }
    gpu.hip
        .memcpy_htod(&io.topk.buf, &topk_bytes(topk))
        .map_err(err)?;
    upload_xs(gpu, io, xs)?;
    for o in [&io.k1, &io.cand] {
        sentinel_fill(gpu, &o.g0, GATE_Y_ALLOC)?;
        sentinel_fill(gpu, &o.g1, GATE_Y_ALLOC)?;
        sentinel_fill(gpu, &o.d0, DOWN_Y_ALLOC)?;
    }
    // Incumbent exactly as dispatch: one plan per layer step, then gate/up, then down.
    k1_plan(gpu, l, mode, &io.topk, lanes)?;
    slab_gate(gpu, l, mode, &io.xg, &io.k1, lanes, SlabK1Module::K1)?;
    slab_down(gpu, l, mode, &io.xd, &io.k1, lanes, SlabK1Module::K1)?;
    match Cand::for_lanes(lanes) {
        Cand::C1 => {
            c1_gate(gpu, l, mode, &io.topk, &io.xg, &io.cand)?;
            c1_down(gpu, l, mode, &io.topk, &io.xd, &io.cand)?;
        }
        Cand::Cn => {
            slab_gate(gpu, l, mode, &io.xg, &io.cand, lanes, SlabK1Module::Cn)?;
            slab_down(gpu, l, mode, &io.xd, &io.cand, lanes, SlabK1Module::Cn)?;
        }
    }
    gpu.hip.device_synchronize().map_err(err)?;
    let slots = lanes * SLAB_TOP_K;
    let mut diff = Diff::default();
    for (part, a, b, alloc, extent, row) in [
        (
            "gate_y0",
            &io.k1.g0,
            &io.cand.g0,
            GATE_Y_ALLOC,
            slots * GATE_Y_ROW,
            GATE_Y_ROW,
        ),
        (
            "gate_y1",
            &io.k1.g1,
            &io.cand.g1,
            GATE_Y_ALLOC,
            slots * GATE_Y_ROW,
            GATE_Y_ROW,
        ),
        (
            "down_y0",
            &io.k1.d0,
            &io.cand.d0,
            DOWN_Y_ALLOC,
            slots * DOWN_Y_ROW,
            DOWN_Y_ROW,
        ),
    ] {
        let (wa, wb) = (
            download_words(gpu, a, alloc)?,
            download_words(gpu, b, alloc)?,
        );
        diff.merge(compare_part(part, &wa, &wb, extent, row, draw, topk));
    }
    Ok(diff)
}

struct CaseRow {
    name: String,
    origin: &'static str,
    layer: Option<usize>,
    mode: Mode,
    lanes: usize,
    draws: usize,
    both_codecs: usize,
    diff: Diff,
}
impl CaseRow {
    fn pass(&self) -> bool {
        self.draws > 0 && self.diff.clean()
    }
    fn json(&self) -> Value {
        let d = &self.diff;
        let mut v = json!({
            "name": self.name, "origin": self.origin, "buckets": self.mode.label(), "lanes": self.lanes,
            "candidate": Cand::for_lanes(self.lanes).name(), "draws": self.draws,
            "words_compared": d.words, "mismatch_count": d.mismatches, "first_mismatches": d.first,
            "k1_unwritten_words": d.k1_unwritten, "cand_unwritten_words": d.cand_unwritten,
            "k1_tail_disturbed_words": d.k1_tail_bad, "cand_tail_disturbed_words": d.cand_tail_bad,
            "draws_with_both_codecs": self.both_codecs, "pass": self.pass(),
        });
        if let Some(l) = self.layer {
            v["layer"] = json!(l);
        }
        v
    }
}

fn both_codecs(l: &Layer, topk: &[i32]) -> bool {
    let w3 = topk.iter().filter(|&&e| l.codec_of(e) == Codec::W3).count();
    w3 > 0 && w3 < topk.len()
}

/// One case at one lanes value: every lane-0 draw of `draws` extended to `lanes` lanes (`lane_routes`).
#[allow(clippy::too_many_arguments)]
fn run_case(
    gpu: &mut Gpu,
    io: &Io,
    l: &Layer,
    mode: Mode,
    name: &str,
    origin: &'static str,
    layer_no: Option<usize>,
    case_idx: usize,
    draws: &[Vec<i32>],
    lanes: usize,
) -> Result<CaseRow> {
    let name = format!("{name}/lanes{lanes}");
    let mut row = CaseRow {
        name: name.clone(),
        origin,
        layer: layer_no,
        mode,
        lanes,
        draws: 0,
        both_codecs: 0,
        diff: Diff::default(),
    };
    let mut rng = Rng::new(name_seed(&name) ^ 0x7a11);
    let mapped = l.ids_of(None);
    for (i, base) in draws.iter().enumerate() {
        let topk = lane_routes(&mut rng, &mapped, base, lanes, i);
        let xs = gen_xs(style_for(origin == "real", case_idx, i), lanes, &mut rng);
        let diff = g1_draw(gpu, l, mode, io, &topk, lanes, &xs, i)?;
        row.draws += 1;
        row.both_codecs += both_codecs(l, &topk) as usize;
        row.diff.merge(diff);
    }
    Ok(row)
}

// ---------------------------------------------------------------------------------------
// Synthetic case list
// ---------------------------------------------------------------------------------------

const SYNTH_CASES: &[&str] = &[
    "syn_uniform_w3",
    "syn_uniform_mq4",
    "syn_mixed_a",
    "syn_mixed_b",
    "syn_mixed_stress_x",
];

/// Pool ids: always 0, 1, 510, 511, then random distinct ids.
fn synth_pool_ids(rng: &mut Rng) -> Vec<u32> {
    let mut ids: BTreeSet<u32> = [0, 1, 510, 511].into_iter().collect();
    while ids.len() < POOL {
        ids.insert(rng.below(SLAB_EXPERTS) as u32);
    }
    let mut v: Vec<u32> = ids.into_iter().collect();
    rng.shuffle(&mut v);
    v
}

fn synth_host_layer(name: &str) -> (HostLayer, Vec<Mode>) {
    let mut rng = Rng::new(name_seed(name) ^ 0x5eed);
    let ids = synth_pool_ids(&mut rng);
    match name {
        "syn_uniform_w3" | "syn_uniform_mq4" => {
            let c = if name == "syn_uniform_w3" {
                Codec::W3
            } else {
                Codec::Mq4
            };
            let pool: Vec<(u32, Codec)> = ids.iter().map(|&i| (i, c)).collect();
            let h = synth_layer(name, &pool, &mut |_| c, &mut rng);
            (h, vec![Mode::Uniform(c.k1_bucket()), Mode::Table])
        }
        _ => {
            // Codecs: "a" random, "b" alternating by position, "stress" random; both >= 8 of each.
            let mut codecs: Vec<Codec> = (0..ids.len())
                .map(|i| match name {
                    "syn_mixed_b" => {
                        if i % 2 == 0 {
                            Codec::W3
                        } else {
                            Codec::Mq4
                        }
                    }
                    _ => {
                        if rng.below(2) == 0 {
                            Codec::W3
                        } else {
                            Codec::Mq4
                        }
                    }
                })
                .collect();
            for c in [Codec::W3, Codec::Mq4] {
                while codecs.iter().filter(|&&x| x == c).count() < 8 {
                    let i = rng.below(codecs.len());
                    codecs[i] = c;
                }
            }
            let pool: Vec<(u32, Codec)> = ids.iter().copied().zip(codecs).collect();
            let h = synth_layer(
                name,
                &pool,
                &mut |r: &mut Rng| {
                    if r.below(2) == 0 {
                        Codec::W3
                    } else {
                        Codec::Mq4
                    }
                },
                &mut rng,
            );
            (h, vec![Mode::Table])
        }
    }
}

// ---------------------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------------------

/// One timed launch sequence. `K1*` are the incumbent (plan + forced K1 buckets); `Cand{Gate,Down,GateDown}` are
/// C1 (`lanes == 1`, no plan); `Cand{PlanGate,PlanGateDown}` are plan + forced Cn (`lanes >= 2`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Seq {
    K1Plan,
    K1PlanGate,
    K1PlanGateDown,
    CandGate,
    CandDown,
    CandGateDown,
    CandPlanGate,
    CandPlanGateDown,
}
fn seqs_for(lanes: usize) -> Vec<Seq> {
    let mut v = vec![Seq::K1Plan, Seq::K1PlanGate, Seq::K1PlanGateDown];
    if lanes == 1 {
        v.extend([Seq::CandGate, Seq::CandDown, Seq::CandGateDown]);
    } else {
        v.extend([Seq::CandPlanGate, Seq::CandPlanGateDown]);
    }
    v
}
impl Seq {
    fn name(self, cand: Cand) -> String {
        let c = cand.name();
        match self {
            Seq::K1Plan => "k1_plan".into(),
            Seq::K1PlanGate => "k1_plan_gate".into(),
            Seq::K1PlanGateDown => "k1_plan_gate_down".into(),
            Seq::CandGate => format!("{c}_gate"),
            Seq::CandDown => format!("{c}_down"),
            Seq::CandGateDown => format!("{c}_gate_down"),
            Seq::CandPlanGate => format!("{c}_plan_gate"),
            Seq::CandPlanGateDown => format!("{c}_plan_gate_down"),
        }
    }
}

fn launch_seq(
    gpu: &mut Gpu,
    s: Seq,
    l: &Layer,
    topk: &GpuTensor,
    io: &Io,
    lanes: usize,
) -> Result<()> {
    let mode = Mode::Table;
    let (k1, cn) = (SlabK1Module::K1, SlabK1Module::Cn);
    match s {
        Seq::K1Plan => k1_plan(gpu, l, mode, topk, lanes),
        Seq::K1PlanGate => {
            k1_plan(gpu, l, mode, topk, lanes)?;
            slab_gate(gpu, l, mode, &io.xg, &io.k1, lanes, k1)
        }
        Seq::K1PlanGateDown => {
            k1_plan(gpu, l, mode, topk, lanes)?;
            slab_gate(gpu, l, mode, &io.xg, &io.k1, lanes, k1)?;
            slab_down(gpu, l, mode, &io.xd, &io.k1, lanes, k1)
        }
        Seq::CandGate => c1_gate(gpu, l, mode, topk, &io.xg, &io.cand),
        Seq::CandDown => c1_down(gpu, l, mode, topk, &io.xd, &io.cand),
        Seq::CandGateDown => {
            c1_gate(gpu, l, mode, topk, &io.xg, &io.cand)?;
            c1_down(gpu, l, mode, topk, &io.xd, &io.cand)
        }
        Seq::CandPlanGate => {
            k1_plan(gpu, l, mode, topk, lanes)?;
            slab_gate(gpu, l, mode, &io.xg, &io.cand, lanes, cn)
        }
        Seq::CandPlanGateDown => {
            k1_plan(gpu, l, mode, topk, lanes)?;
            slab_gate(gpu, l, mode, &io.xg, &io.cand, lanes, cn)?;
            slab_down(gpu, l, mode, &io.xd, &io.cand, lanes, cn)
        }
    }
}

struct Draw {
    layer: usize,
    host: Vec<i32>,
    dev: GpuTensor,
}

/// `(gate_bytes, down_bytes)` of the DISTINCT routed experts of `topk` (an expert repeated across lanes counts once).
fn route_bytes(l: &Layer, topk: &[i32]) -> (u64, u64) {
    let distinct: BTreeSet<i32> = topk.iter().copied().collect();
    distinct.iter().fold((0, 0), |(g, d), &e| {
        (g + l.gate_bytes(e) as u64, d + l.down_bytes(e) as u64)
    })
}

/// Unique `(layer, expert)` weight bytes (gate + down) touched by `draws[start .. start + n)` (cyclic).
fn window_unique_bytes(layers: &[Layer], draws: &[Draw], start: usize, n: usize) -> u64 {
    let mut seen = BTreeSet::new();
    let mut bytes = 0u64;
    for i in 0..n {
        let d = &draws[(start + i) % draws.len()];
        for &e in &d.host {
            if seen.insert((d.layer, e)) {
                let l = &layers[d.layer];
                bytes += (l.gate_bytes(e) + l.down_bytes(e)) as u64;
            }
        }
    }
    bytes
}

struct TimingCfg {
    reps: usize,
    warmup: usize,
    launches: usize,
}

/// Time the sequences of one `lanes` value over `draws` (routes of `lanes` lanes): the per-lanes JSON entry.
fn timing_run(
    gpu: &mut Gpu,
    layers: &[Layer],
    draws: &[Draw],
    io: &Io,
    t: &TimingCfg,
    lanes: usize,
) -> Result<Value> {
    let cand = Cand::for_lanes(lanes);
    let seqs = seqs_for(lanes);
    let start = gpu.hip.event_create().map_err(err)?;
    let stop = gpu.hip.event_create().map_err(err)?;
    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); seqs.len()];
    for rep in 0..t.warmup + t.reps {
        let cursor = rep * t.launches;
        for k in 0..seqs.len() {
            let si = (k + rep) % seqs.len();
            let seq = seqs[si];
            gpu.hip
                .event_record(&start, gpu.active_stream.as_ref())
                .map_err(err)?;
            for i in 0..t.launches {
                let d = &draws[(cursor + i) % draws.len()];
                launch_seq(gpu, seq, &layers[d.layer], &d.dev, io, lanes)?;
            }
            gpu.hip
                .event_record(&stop, gpu.active_stream.as_ref())
                .map_err(err)?;
            gpu.hip.event_synchronize(&stop).map_err(err)?;
            let ms = gpu.hip.event_elapsed_ms(&start, &stop).map_err(err)?;
            if rep >= t.warmup {
                samples[si].push(ms as f64 * 1000.0 / t.launches as f64);
            }
        }
    }
    let _ = gpu.hip.event_destroy(start);
    let _ = gpu.hip.event_destroy(stop);

    let med = |s: Seq| median(&samples[seqs.iter().position(|&x| x == s).unwrap()]);
    let (plan, pg, pgd) = (
        med(Seq::K1Plan),
        med(Seq::K1PlanGate),
        med(Seq::K1PlanGateDown),
    );
    // Candidate gate/up, down and gate+down: C1 measures the projections directly, Cn derives down from
    // the plan-inclusive sequences exactly as K1's down is derived.
    let (cand_gate, cand_down, cand_total) = if lanes == 1 {
        (
            med(Seq::CandGate),
            med(Seq::CandDown),
            med(Seq::CandGateDown),
        )
    } else {
        let (cpg, cpgd) = (med(Seq::CandPlanGate), med(Seq::CandPlanGateDown));
        (cpg, cpgd - cpg, cpgd)
    };
    let k1_gate = pg;
    let k1_down = pgd - pg;
    let (mut gate_b, mut down_b) = (0u64, 0u64);
    for d in draws {
        let (g, dn) = route_bytes(&layers[d.layer], &d.host);
        gate_b += g;
        down_b += dn;
    }
    let (gate_mean, down_mean) = (
        gate_b as f64 / draws.len() as f64,
        down_b as f64 / draws.len() as f64,
    );
    let gbps = |bytes: f64, us: f64| bytes / (us * 1e-6) / 1e9;
    let window_min = (0..t.reps)
        .map(|r| window_unique_bytes(layers, draws, (t.warmup + r) * t.launches, t.launches))
        .min()
        .unwrap_or(0);
    let pool_unique = window_unique_bytes(layers, draws, 0, draws.len());
    let proj = |bytes: f64, k1_us: f64, cand_us: f64| {
        json!({"unique_bytes_mean": bytes, "k1_us": k1_us, "cand_us": cand_us,
            "k1_gbps": gbps(bytes, k1_us), "cand_gbps": gbps(bytes, cand_us), "cand_speedup": k1_us / cand_us})
    };
    let names: Vec<String> = seqs.iter().map(|s| s.name(cand)).collect();
    let raw: serde_json::Map<String, Value> = names
        .iter()
        .zip(&samples)
        .map(|(n, s)| (n.clone(), json!(s)))
        .collect();
    let medians: serde_json::Map<String, Value> = names
        .iter()
        .zip(&samples)
        .map(|(n, s)| (n.clone(), json!(median(s))))
        .collect();
    let cand_down_is = if lanes == 1 {
        format!("median({}_down) (a standalone launch)", cand.name())
    } else {
        format!("median({0}_plan_gate_down) - median({0}_plan_gate) (derived; the union plan is shared)", cand.name())
    };
    Ok(json!({
        "lanes": lanes, "candidate": cand.name(),
        "draws": draws.len(),
        "draws_per_sample": t.launches.min(draws.len()),
        "pool_unique_weight_bytes": pool_unique,
        "min_sample_unique_weight_bytes": window_min,
        "sample_working_set_exceeds_mall": window_min > MALL_BYTES,
        "plan_us": plan,
        "median_us": Value::Object(medians),
        "projections": {
            "gate_up": proj(gate_mean, k1_gate, cand_gate),
            "down": proj(down_mean, k1_down, cand_down),
            "k1_down_is": "median(k1_plan_gate_down) - median(k1_plan_gate) (derived; the union plan is shared)",
            "cand_down_is": cand_down_is,
            "gate_up_includes_plan": "k1 always; the candidate only at lanes >= 2 (it shares the union plan)",
        },
        "gate_down_total": {"k1_us": pgd, "cand_us": cand_total, "cand_speedup": pgd / cand_total,
            "unique_bytes_mean": gate_mean + down_mean,
            "k1_gbps": gbps(gate_mean + down_mean, pgd), "cand_gbps": gbps(gate_mean + down_mean, cand_total)},
        "samples_us": Value::Object(raw),
    }))
}

// ---------------------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------------------

#[derive(Debug)]
struct Cfg {
    artifact: Option<String>,
    layers: Vec<usize>,
    draws: Option<usize>,
    hot_frac: Option<f64>,
    seed: u64,
    skip_synthetic: bool,
    only: Vec<String>,
    timing: bool,
    reps: usize,
    warmup: usize,
    launches: usize,
    lanes: Vec<usize>,
    device: Option<i32>,
    expect_arch: String,
    expect_uuid: Option<String>,
    expect_bdf: Option<String>,
    card: Option<String>,
    fabric_card: Option<String>,
    fabric_owner: Option<String>,
    fabric_wrapper: Option<String>,
}

const USAGE: &str = "usage: mq_i4_moe_gemv_c1_oracle [--artifact SLAB.mq3-xts --layers L,L,..] [--draws N(>=64)] \
[--hot-frac F] [--lanes L,L,..(1..=8, default 1,2,3,4,5,8)] [--seed N] [--skip-synthetic] [--only SUBSTR]... \
[--timing --reps N(>=20) --warmup N(>=3) --launches N(>=64)] [--device N] [--expect-arch A] [--expect-uuid U] \
[--expect-bdf B] [--card NAME] [--fabric-owner OWNER] [--fabric-wrapper PATH] [--fabric-card NAME]";

fn env_nonempty(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

fn parse_args(argv: &[String]) -> Result<Cfg> {
    let mut cfg = Cfg {
        artifact: None,
        layers: Vec::new(),
        draws: None,
        hot_frac: None,
        seed: 0,
        skip_synthetic: false,
        only: Vec::new(),
        timing: false,
        reps: 30,
        warmup: 5,
        launches: 64,
        lanes: DEFAULT_LANES.to_vec(),
        device: None,
        expect_arch: DEFAULT_ARCH.into(),
        expect_uuid: None,
        expect_bdf: None,
        card: env_nonempty("FABRIC_CARD"),
        fabric_card: None,
        fabric_owner: env_nonempty("FABRIC_OWNER"),
        fabric_wrapper: env_nonempty("FABRIC_WRAPPER"),
    };
    let mut i = 0;
    let value = |i: &mut usize| -> Result<String> {
        *i += 1;
        argv.get(*i)
            .cloned()
            .ok_or_else(|| format!("{} needs a value\n{USAGE}", argv[*i - 1]))
    };
    while i < argv.len() {
        let flag = argv[i].clone();
        let num = |s: String| s.parse::<usize>().map_err(|e| format!("{flag}: {e}"));
        match flag.as_str() {
            "--artifact" => cfg.artifact = Some(value(&mut i)?),
            "--layers" => {
                let v = value(&mut i)?;
                cfg.layers = v
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(|s| {
                        s.trim()
                            .parse::<usize>()
                            .map_err(|e| format!("--layers {s}: {e}"))
                    })
                    .collect::<Result<Vec<_>>>()?;
            }
            "--draws" => cfg.draws = Some(num(value(&mut i)?)?),
            "--hot-frac" => {
                let f: f64 = value(&mut i)?
                    .parse()
                    .map_err(|e| format!("--hot-frac: {e}"))?;
                if !(0.0..=1.0).contains(&f) {
                    return Err("--hot-frac must be in [0, 1]".into());
                }
                cfg.hot_frac = Some(f);
            }
            "--seed" => cfg.seed = num(value(&mut i)?)? as u64,
            "--skip-synthetic" => cfg.skip_synthetic = true,
            "--only" => cfg.only.push(value(&mut i)?),
            "--timing" => cfg.timing = true,
            "--reps" => cfg.reps = num(value(&mut i)?)?,
            "--warmup" => cfg.warmup = num(value(&mut i)?)?,
            "--launches" => cfg.launches = num(value(&mut i)?)?,
            "--lanes" => {
                let v = value(&mut i)?;
                let lanes = v
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(|s| {
                        s.trim()
                            .parse::<usize>()
                            .map_err(|e| format!("--lanes {s}: {e}"))
                    })
                    .collect::<Result<Vec<_>>>()?;
                if lanes.is_empty()
                    || lanes
                        .iter()
                        .any(|&l| !(1..=SLAB_NARROW_MAX_LANES).contains(&l))
                {
                    return Err(format!(
                        "--lanes takes values in 1..={SLAB_NARROW_MAX_LANES}\n{USAGE}"
                    ));
                }
                if lanes
                    .iter()
                    .enumerate()
                    .any(|(i, l)| lanes[..i].contains(l))
                {
                    return Err("--lanes must not repeat a value".into());
                }
                cfg.lanes = lanes;
            }
            "--device" => cfg.device = Some(num(value(&mut i)?)? as i32),
            "--expect-arch" => cfg.expect_arch = value(&mut i)?,
            "--expect-uuid" => cfg.expect_uuid = Some(value(&mut i)?),
            "--expect-bdf" => cfg.expect_bdf = Some(value(&mut i)?.to_ascii_lowercase()),
            "--card" => cfg.card = Some(value(&mut i)?),
            "--fabric-card" => cfg.fabric_card = Some(value(&mut i)?),
            "--fabric-owner" => cfg.fabric_owner = Some(value(&mut i)?),
            "--fabric-wrapper" => cfg.fabric_wrapper = Some(value(&mut i)?),
            other => return Err(format!("unknown argument {other}\n{USAGE}")),
        }
        i += 1;
    }
    if cfg.artifact.is_some() != !cfg.layers.is_empty() {
        return Err(format!("--artifact and --layers go together\n{USAGE}"));
    }
    if cfg.timing {
        if cfg.artifact.is_none() {
            return Err("--timing needs --artifact and --layers (real layers)".into());
        }
        if cfg.reps < 20 || cfg.warmup < 3 {
            return Err("--timing requires --reps >= 20 and --warmup >= 3".into());
        }
        if cfg.launches < MIN_DRAWS {
            return Err(format!(
                "--timing requires --launches >= {MIN_DRAWS} (distinct routes rotate per sample)"
            ));
        }
    }
    let draws = cfg
        .draws
        .unwrap_or(if cfg.timing { 256 } else { MIN_DRAWS });
    if draws < MIN_DRAWS {
        return Err(format!("--draws must be >= {MIN_DRAWS}"));
    }
    cfg.draws = Some(draws);
    Ok(cfg)
}

fn selected(cfg: &Cfg, name: &str) -> bool {
    cfg.only.is_empty() || cfg.only.iter().any(|s| name.contains(s.as_str()))
}

fn module_json(module: &str, symbols: &[&str], image: &[u8]) -> Value {
    json!({"module": module, "symbols": symbols, "sha256": sha_of(image), "bytes": image.len()})
}

fn run() -> Result<(Value, bool)> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let cfg = parse_args(&argv)?;
    let mut gpu = Gpu::init_with_device(cfg.device.unwrap_or(0)).map_err(err)?;
    let id = gpu.device_id;
    let uuid = gpu.hip.device_uuid(id).map_err(err)?;
    let bdf = gpu
        .hip
        .device_pci_bus_id(id)
        .map_err(err)?
        .to_ascii_lowercase();
    let arch = gpu.hip.get_arch(id).map_err(err)?;
    let identity_match = arch == cfg.expect_arch
        && gpu.arch == cfg.expect_arch
        && cfg.expect_uuid.as_ref().is_none_or(|u| *u == uuid)
        && cfg.expect_bdf.as_ref().is_none_or(|b| *b == bdf);
    let device = json!({"index": id, "uuid": uuid, "bdf": bdf, "arch": arch, "card": cfg.card,
        "expected_uuid": cfg.expect_uuid, "expected_bdf": cfg.expect_bdf, "expected_arch": cfg.expect_arch,
        "identity_match": identity_match});
    let command = json!({"argv": argv, "build_command": BUILD_COMMAND});
    let modules = json!({
        "k1": module_json(K1_MODULE, &MQ_I4_MOE_GEMV_SYMBOLS.iter().flatten().copied().collect::<Vec<_>>(),
            kernels::MQ_I4_MOE_GEMV_PM_GFX1151),
        "c1": module_json(C1_MODULE, &MQ_I4_MOE_GEMV_C1_SYMBOLS, kernels::MQ_I4_MOE_GEMV_C1_PM_GFX1151),
        "cn": module_json(CN_MODULE, &MQ_I4_MOE_GEMV_CN_SYMBOLS.iter().flatten().copied().collect::<Vec<_>>(),
            kernels::MQ_I4_MOE_GEMV_CN_PM_GFX1151),
    });
    if !identity_match || gpu.arch != DEFAULT_ARCH {
        return Ok((
            json!({"schema": SCHEMA, "command": command, "device": device, "modules": modules,
                "error": "K1-C1/Cn oracle device identity mismatch (gfx1151 only)", "byte_exact": false, "pass": false, "rc": 1}),
            false,
        ));
    }
    gpu.ensure_qwen4_slab_kernels(COLD_W3).map_err(err)?;
    gpu.reserve_qwen4_slab(0).map_err(err)?;
    let io = alloc_io(&mut gpu)?;
    let hot_frac = cfg.hot_frac;
    let draws_n = cfg.draws.unwrap();

    let mut rows: Vec<CaseRow> = Vec::new();
    let mut artifact_meta = Value::Null;
    let mut timing = Value::Null;

    if !cfg.timing {
        if !cfg.skip_synthetic {
            for (ci, name) in SYNTH_CASES.iter().filter(|n| selected(&cfg, n)).enumerate() {
                eprintln!("[g1] {name}");
                let (host, modes) = synth_host_layer(name);
                let layer = upload_layer(&mut gpu, name, host)?;
                let mut rng = Rng::new(name_seed(name) ^ cfg.seed ^ 0xd2a3);
                let draws = gen_draws(&mut rng, &layer, SYNTH_DRAWS, None, true);
                let multi = modes.len() > 1;
                for mode in modes {
                    let label = if multi {
                        format!("{name}/{}", mode.label())
                    } else {
                        name.to_string()
                    };
                    for &lanes in &cfg.lanes {
                        rows.push(run_case(
                            &mut gpu,
                            &io,
                            &layer,
                            mode,
                            &label,
                            "synthetic",
                            None,
                            ci,
                            &draws,
                            lanes,
                        )?);
                    }
                }
                free_layer(&mut gpu, layer)?;
            }
        }
        if let Some(path) = &cfg.artifact {
            let art = open_artifact(path)?;
            let mut metas = Vec::new();
            for &ln in &cfg.layers {
                let name = format!("real_L{ln}");
                if !selected(&cfg, &name) {
                    continue;
                }
                eprintln!("[g1] loading layer {ln} of {path}");
                let (host, meta) = load_real_layer(&art, ln)?;
                metas.push(meta);
                let layer = upload_layer(&mut gpu, &name, host)?;
                let mut rng = Rng::new(mix64(ln as u64) ^ cfg.seed ^ 0x7ea1);
                let draws = gen_draws(&mut rng, &layer, draws_n, hot_frac, true);
                let mut modes = vec![Mode::Table];
                let (hot, cold) = (
                    layer.ids_of(Some(Codec::Mq4)).len(),
                    layer.ids_of(Some(Codec::W3)).len(),
                );
                if hot == 0 {
                    modes.push(Mode::Uniform(Codec::Mq4.k1_bucket()));
                } else if cold == 0 {
                    modes.push(Mode::Uniform(Codec::W3.k1_bucket()));
                }
                for mode in modes {
                    eprintln!(
                        "[g1] {name} {} ({} draws x lanes {:?})",
                        mode.label(),
                        draws.len(),
                        cfg.lanes
                    );
                    let label = if mode == Mode::Table {
                        name.clone()
                    } else {
                        format!("{name}/{}", mode.label())
                    };
                    for &lanes in &cfg.lanes {
                        rows.push(run_case(
                            &mut gpu,
                            &io,
                            &layer,
                            mode,
                            &label,
                            "real",
                            Some(ln),
                            0,
                            &draws,
                            lanes,
                        )?);
                    }
                }
                free_layer(&mut gpu, layer)?;
            }
            artifact_meta = json!({"path": path, "file_len": art.hfq.file_len(),
                "index_sha256": hex(&art.hfq.index_sha256().map_err(err)?), "layers": metas});
        }
    } else {
        let path = cfg.artifact.as_ref().unwrap();
        let art = open_artifact(path)?;
        let mut layers: Vec<Layer> = Vec::new();
        let mut metas = Vec::new();
        for &ln in &cfg.layers {
            eprintln!("[timing] loading layer {ln} of {path}");
            let (host, meta) = load_real_layer(&art, ln)?;
            metas.push(meta);
            layers.push(upload_layer(&mut gpu, &format!("real_L{ln}"), host)?);
        }
        artifact_meta = json!({"path": path, "file_len": art.hfq.file_len(),
            "index_sha256": hex(&art.hfq.index_sha256().map_err(err)?), "layers": metas});
        let per_layer = draws_n.div_ceil(layers.len());
        let mut per: Vec<Vec<Vec<i32>>> = Vec::new();
        for (li, l) in layers.iter().enumerate() {
            let mut rng = Rng::new(mix64(cfg.layers[li] as u64) ^ cfg.seed ^ 0x71a1);
            per.push(gen_draws(&mut rng, l, per_layer, hot_frac, false));
        }
        let tcfg = TimingCfg {
            reps: cfg.reps,
            warmup: cfg.warmup,
            launches: cfg.launches,
        };
        let mapped: Vec<Vec<u32>> = layers.iter().map(|l| l.ids_of(None)).collect();
        let mut by_lanes: Vec<Value> = Vec::new();
        for &lanes in &cfg.lanes {
            let cand = Cand::for_lanes(lanes);
            let mut route_rng = Rng::new(mix64(lanes as u64) ^ cfg.seed ^ 0x71a2);
            let mut draws: Vec<Draw> = Vec::new();
            for d in 0..draws_n {
                let layer = d % layers.len();
                let host = lane_routes(
                    &mut route_rng,
                    &mapped[layer],
                    &per[layer][d / layers.len()],
                    lanes,
                    d,
                );
                let dev = up_raw(&mut gpu, &topk_bytes(&host))?;
                draws.push(Draw { layer, host, dev });
            }
            // Refuse to time a wrong kernel: G1 over the first draws at this lanes value, through the same tables.
            let mut rng = Rng::new(cfg.seed ^ 0x71e0 ^ mix64(lanes as u64));
            let mut verify = Diff::default();
            let n_verify = TIMING_VERIFY_DRAWS.min(draws.len());
            for (i, d) in draws.iter().take(n_verify).enumerate() {
                let xs = gen_xs(style_for(true, 0, i), lanes, &mut rng);
                let diff = g1_draw(
                    &mut gpu,
                    &layers[d.layer],
                    Mode::Table,
                    &io,
                    &d.host,
                    lanes,
                    &xs,
                    i,
                )?;
                verify.merge(diff);
            }
            let precheck = CaseRow {
                name: format!("timing_precheck/lanes{lanes}"),
                origin: "real",
                layer: None,
                mode: Mode::Table,
                lanes,
                draws: n_verify,
                both_codecs: draws
                    .iter()
                    .take(n_verify)
                    .filter(|d| both_codecs(&layers[d.layer], &d.host))
                    .count(),
                diff: verify,
            };
            let ok = precheck.pass();
            rows.push(precheck);
            if !ok {
                return Err(format!(
                    "K1-{} is not byte-exact against K1 at {lanes} lanes on the timing draws; refusing to time a wrong kernel",
                    cand.name().to_ascii_uppercase()
                ));
            }
            // Timing inputs: one realistic X (the value path does not change the memory schedule).
            let mut xr = Rng::new(cfg.seed ^ 0x71e1);
            upload_xs(&gpu, &io, &gen_xs(0, lanes, &mut xr))?;
            eprintln!(
                "[timing] lanes {lanes} ({}): {} draws over {} layers, {} reps",
                cand.name(),
                draws.len(),
                layers.len(),
                tcfg.reps
            );
            by_lanes.push(timing_run(&mut gpu, &layers, &draws, &io, &tcfg, lanes)?);
            for d in draws {
                gpu.free_tensor(d.dev).map_err(err)?;
            }
        }
        let card = cfg.card.clone().unwrap_or_default();
        timing = json!({
            "event_timing": "hipEventElapsedTime over `launches_per_sample` back-to-back launches of consecutive draws, us per launch",
            "layers": layers.iter().map(|l| l.name.clone()).collect::<Vec<_>>(),
            "draws": draws_n, "reps": tcfg.reps, "warmup": tcfg.warmup, "launches_per_sample": tcfg.launches,
            "mall_bytes": MALL_BYTES,
            "fabric": {"class": "timing", "card": cfg.fabric_card.clone().unwrap_or(card),
                "owner": cfg.fabric_owner, "wrapper": cfg.fabric_wrapper},
            "timing_source": "device",
            "verified_byte_exact_before_timing": true,
            "by_lanes": by_lanes,
        });
        for l in layers {
            free_layer(&mut gpu, l)?;
        }
    }
    free_io(&mut gpu, io)?;

    let mismatches: u64 = rows.iter().map(|r| r.diff.mismatches).sum();
    let byte_exact = !rows.is_empty() && rows.iter().all(|r| r.pass());
    let count = |origin: &str| rows.iter().filter(|r| r.origin == origin).count();
    let mut out = json!({
        "schema": SCHEMA,
        "mode": if cfg.timing { "timing" } else { "g1" },
        "command": command,
        "device": device,
        "modules": modules,
        "config": {"layers": cfg.layers, "draws": draws_n, "hot_frac": cfg.hot_frac, "seed": cfg.seed,
            "only": cfg.only, "skip_synthetic": cfg.skip_synthetic, "timing": cfg.timing, "reps": cfg.reps,
            "warmup": cfg.warmup, "launches_per_sample": cfg.launches, "lanes": cfg.lanes,
            "sentinel": bits_hex(SENTINEL_BITS)},
        "cases": rows.iter().map(CaseRow::json).collect::<Vec<_>>(),
        "case_counts": {"total": rows.len(), "synthetic": count("synthetic"), "real": count("real"),
            "draws": rows.iter().map(|r| r.draws).sum::<usize>()},
        "mismatch_count": mismatches,
        "byte_exact": byte_exact,
        "pass": byte_exact,
        "rc": if byte_exact { 0 } else { 1 },
    });
    if !artifact_meta.is_null() {
        out["artifact"] = artifact_meta;
    }
    if !timing.is_null() {
        out["timing"] = timing;
    }
    Ok((out, byte_exact))
}

fn main() {
    match run() {
        Ok((v, pass)) => {
            println!("{v}");
            if !pass {
                eprintln!("MQ-I4 K1-C1/Cn oracle FAILED");
                std::process::exit(1);
            }
        }
        Err(e) => {
            println!(
                "{}",
                json!({"schema": SCHEMA, "error": e, "byte_exact": false, "pass": false, "rc": 1})
            );
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}

// ---------------------------------------------------------------------------------------
// Local known-answer tests (no GPU):
//   cargo test -p hipfire-runtime --release --features lab --example mq_i4_moe_gemv_c1_oracle
// ---------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_tensor_sizes_are_the_slab_payload_sizes() {
        let mut rng = Rng::new(1);
        for c in [Codec::W3, Codec::Mq4] {
            assert_eq!(
                gen_tensor(c, false, 0, true, &mut rng).len(),
                c.format().gate_up_bytes()
            );
            assert_eq!(
                gen_tensor(c, true, 1, false, &mut rng).len(),
                c.format().down_bytes()
            );
        }
    }

    #[test]
    fn mq4_headers_are_symmetric_and_finite() {
        let mut rng = Rng::new(2);
        let t = gen_tensor(Codec::Mq4, false, 0, true, &mut rng);
        for g in t.chunks_exact(136) {
            for h in [0usize, 4] {
                let sc = u16::from_le_bytes([g[h], g[h + 1]]);
                let z = u16::from_le_bytes([g[h + 2], g[h + 3]]);
                assert!(sc & 0x7c00 != 0x7c00);
                let z_exp = -8.0 * f16f(sc);
                assert!(z_exp.abs() <= 65504.0, "sc {sc:#06x}");
                assert_eq!(f16f(z), z_exp, "sc {sc:#06x}");
            }
        }
        let t = gen_tensor(Codec::Mq4, true, 0, true, &mut rng);
        assert!(t
            .chunks_exact(68)
            .all(|g| u16::from_le_bytes([g[0], g[1]]) & 0x7c00 != 0x7c00));
    }

    #[test]
    fn w3_scale_classes_cover_the_edges_and_stay_finite() {
        let mut rng = Rng::new(3);
        let t = gen_tensor(Codec::W3, true, 0, true, &mut rng);
        let plane = SLAB_DOWN_M * SLAB_DOWN_K * 3 / 8;
        let scales: BTreeSet<u16> = t[plane..]
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect();
        for edge in [
            0x0000u16, 0x8000, 0x0001, 0x8001, 0x0400, 0x8400, 0x7bff, 0xfbff, 0x3c00,
        ] {
            assert!(scales.contains(&edge), "missing scale {edge:#06x}");
        }
        assert!(scales.iter().all(|s| s & 0x7c00 != 0x7c00));
    }

    #[test]
    fn x_styles_hit_zeros_subnormals_and_large_values() {
        let mut rng = Rng::new(4);
        let z = gen_x_row(3, 64, &mut rng);
        assert!(z.iter().any(|v| v.to_bits() == 0) && z.iter().any(|v| v.to_bits() == 0x8000_0000));
        let s = gen_x_row(4, 256, &mut rng);
        assert!(s.iter().any(|v| *v != 0.0 && v.abs() < f32::MIN_POSITIVE));
        let big = gen_x_row(5, 64, &mut rng);
        assert!(big.iter().all(|v| v.abs() >= 1.0e30));
    }

    #[test]
    fn synthetic_layers_place_experts_256_aligned_with_a_zero_tail() {
        let (h, modes) = synth_host_layer("syn_mixed_a");
        assert_eq!(modes, vec![Mode::Table]);
        let mapped: Vec<usize> = (0..SLAB_EXPERTS)
            .filter(|&i| h.gate_off[i] != UNMAPPED)
            .collect();
        assert_eq!(mapped.len(), POOL);
        for &i in &mapped {
            assert_eq!(h.gate_off[i] % ARENA_ALIGN, 0);
            let c = Codec::from_byte(h.codec[i]);
            assert_eq!(h.down_off[i], h.gate_off[i] + c.format().gate_up_bytes());
            assert!(
                h.down_off[i] + c.format().down_bytes() + SLAB_ARENA_TAIL_PAD_BYTES
                    <= h.arena.len()
            );
        }
        assert!(h.arena[h.arena.len() - SLAB_ARENA_TAIL_PAD_BYTES..]
            .iter()
            .all(|&b| b == 0));
        let w3 = mapped.iter().filter(|&&i| h.codec[i] == CODEC_W3).count();
        assert!(w3 >= 8 && mapped.len() - w3 >= 8);
        assert!(mapped.contains(&0) && mapped.contains(&511));
    }

    #[test]
    fn compare_is_bitwise_and_checks_the_sentinel_tail() {
        let n = 8;
        let extent = 6;
        let mut a = vec![0x3f80_0000u32; n];
        a[6] = SENTINEL_BITS;
        a[7] = SENTINEL_BITS;
        let b = a.clone();
        assert!(compare_part("p", &a, &b, extent, 3, 0, &[0]).clean());
        // +0 vs -0 and NaN payloads differ.
        let mut c = a.clone();
        c[0] = 0x8000_0000;
        a[0] = 0;
        let d = compare_part("p", &a, &c, extent, 3, 0, &[0]);
        assert_eq!(d.mismatches, 1);
        assert_eq!(d.first[0]["slot"], 0);
        // A write into the tail and an unwritten word in the extent are failures even when equal.
        let mut e = a.clone();
        e[7] = 0;
        let d = compare_part("p", &e, &e, extent, 3, 0, &[0]);
        assert_eq!((d.k1_tail_bad, d.cand_tail_bad), (1, 1));
        let mut f = a.clone();
        f[2] = SENTINEL_BITS;
        let d = compare_part("p", &f, &f, extent, 3, 0, &[0]);
        assert_eq!((d.k1_unwritten, d.cand_unwritten), (1, 1));
        assert!(!d.clean());
    }

    #[test]
    fn median_and_window_math() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 2.0, 3.0]), 2.5);
        assert!(median(&[]).is_nan());
    }

    #[test]
    fn args_enforce_timing_minimums_and_pairing() {
        let a = |s: &[&str]| parse_args(&s.iter().map(|x| x.to_string()).collect::<Vec<_>>());
        assert!(a(&[]).is_ok());
        assert!(a(&["--artifact", "/x"]).is_err());
        assert!(a(&["--timing"]).is_err());
        assert!(a(&[
            "--timing",
            "--artifact",
            "/x",
            "--layers",
            "0",
            "--launches",
            "32"
        ])
        .is_err());
        assert!(a(&["--timing", "--artifact", "/x", "--layers", "0"]).is_ok());
        assert!(a(&["--draws", "63"]).is_err());
        assert_eq!(
            a(&["--artifact", "/x", "--layers", "0,1,23"])
                .unwrap()
                .layers,
            vec![0, 1, 23]
        );
    }

    #[test]
    fn lanes_flag_defaults_and_validates() {
        let a = |s: &[&str]| parse_args(&s.iter().map(|x| x.to_string()).collect::<Vec<_>>());
        assert_eq!(a(&[]).unwrap().lanes, vec![1, 2, 3, 4, 5, 8]);
        assert_eq!(a(&["--lanes", "2,8,1"]).unwrap().lanes, vec![2, 8, 1]);
        assert_eq!(a(&["--lanes", "1,2,3,4,5,6,7,8"]).unwrap().lanes.len(), 8);
        assert!(a(&["--lanes", "0"]).is_err());
        assert!(a(&["--lanes", "9"]).is_err());
        assert!(a(&["--lanes", "2,2"]).is_err());
        assert!(a(&["--lanes", ""]).is_err());
    }

    #[test]
    fn lane_routes_are_distinct_per_lane_and_overlap_lane_zero() {
        let all: Vec<u32> = (0..SLAB_EXPERTS as u32).collect();
        let base: Vec<i32> = (100..100 + SLAB_TOP_K as i32).collect();
        let mut rng = Rng::new(9);
        for draw in 0..8 {
            let r = lane_routes(&mut rng, &all, &base, SLAB_NARROW_MAX_LANES, draw);
            assert_eq!(r.len(), SLAB_NARROW_MAX_LANES * SLAB_TOP_K);
            assert_eq!(&r[..SLAB_TOP_K], &base[..]);
            for lane in 0..SLAB_NARROW_MAX_LANES {
                let route = &r[lane * SLAB_TOP_K..(lane + 1) * SLAB_TOP_K];
                let set: BTreeSet<i32> = route.iter().copied().collect();
                assert_eq!(set.len(), SLAB_TOP_K, "lane {lane} repeats an expert");
                if lane > 0 {
                    let shared = route.iter().filter(|e| base.contains(e)).count();
                    assert!(
                        shared >= lane_keep(draw, lane),
                        "draw {draw} lane {lane}: {shared} shared"
                    );
                }
            }
        }
        // One lane is exactly the base route; draw % 4 == 1 makes every lane a permutation of it.
        assert_eq!(lane_routes(&mut rng, &all, &base, 1, 0), base);
        let perm = lane_routes(&mut rng, &all, &base, 4, 1);
        for lane in 1..4 {
            let mut v = perm[lane * SLAB_TOP_K..(lane + 1) * SLAB_TOP_K].to_vec();
            v.sort_unstable();
            assert_eq!(v, base);
        }
    }

    #[test]
    fn multi_lane_x_shapes_and_candidates() {
        let mut rng = Rng::new(5);
        for lanes in [1usize, 2, 8] {
            let xs = gen_xs(0, lanes, &mut rng);
            assert_eq!(xs.gate.len(), lanes * SLAB_GATE_UP_K);
            assert_eq!(xs.down.len(), lanes * SLAB_TOP_K * SLAB_DOWN_K);
        }
        assert_eq!(Cand::for_lanes(1), Cand::C1);
        assert!((2..=SLAB_NARROW_MAX_LANES).all(|l| Cand::for_lanes(l) == Cand::Cn));
        assert_eq!(seqs_for(1).len(), 6);
        assert_eq!(seqs_for(2).len(), 5);
        assert_eq!(Seq::CandPlanGateDown.name(Cand::Cn), "cn_plan_gate_down");
        assert_eq!(Seq::CandGate.name(Cand::C1), "c1_gate");
    }
}
