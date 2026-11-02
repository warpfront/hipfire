// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! K-E2 E8P-on-IU4 pointer-addressed MoE decode GEMV oracle + row-time runner, gfx1201 and gfx1151, raw PM module (never runtime
//! dispatch except the deliberate `Gpu::e8p_i4_moe_gemv_slab` scratch-path check).
//!
//! Subject: `e8p_i4_moe_gemv_pm_gfx1201` (`e8p_i4_moe_gemv_r256_gate_up`, `e8p_i4_moe_gemv_r128_down`) or, with `--expect-arch gfx1151`,
//! `e8p_i4_moe_gemv_pm_gfx1151` (`e8p_i4_moe_gemv_{r256_gate_up,r128_down}_pm_gfx1151`), launched through the
//! loader-independent launcher `rdna_compute::e8p_moe_decode` (frozen F4 96-byte kernarg, grid `[M, U, 1]`, block 32). Every
//! F32 output word (all slots, every row, padded strides and trailing guard words) is compared **bit for bit** against the
//! frozen K-E0 lane-pair twin of `hipfire_runtime::e8p_i4_reference`: `decode_weights_q` (the lane-pair decode of the wire
//! words) + `mq_i4_reference::gemv_tree_f32_q` with the per-group scale `f32(d)*0.25` (exactly the body of
//! `decode_gemv_f32_lanepair`, which is additionally evaluated directly for the first slot of every launch). No tolerance,
//! no portable-math fallback; the GEMV has no epilogue, so there is no device probe object.
//!
//! Build (the parent builds; this file never builds or runs itself):
//!   cargo build -p hipfire-runtime --release --features lab --example e8p_i4_moe_gemv_oracle
//!
//! CLI (`--e8p-codeobject` is the emitted `.hxaco` bundle or a bare `.co`; GPU runs only on a fabric-granted card):
//!   e8p_i4_moe_gemv_oracle synthetic --e8p-codeobject OBJ --out RECEIPT.json [--seed N] [--only SUBSTR]...
//!   e8p_i4_moe_gemv_oracle real      --e8p-codeobject OBJ --out RECEIPT.json --records-dir DIR --manifest FILE [--layer 0]
//!   e8p_i4_moe_gemv_oracle timing    --e8p-codeobject OBJ --mq4-codeobject MQ4.hxaco --out RECEIPT.json
//!       [--records-dir DIR --manifest FILE] [--reps 25] [--warmup 5] [--batch 64] [--evict-mb 256] [--pool-experts 512]
//!       [--modes resident,cold] [--cs 1,2,3,4,5,6,7,8] [--allow-few-reps]
//!   common: [--device N] [--card LABEL] [--expect-uuid GPU-..|HEX] [--expect-bdf 0000:..] [--expect-arch gfx1201|gfx1151]
//!       [--e8p-codeobject-sha256 HEX] [--mq4-codeobject-sha256 HEX] [--negative-control] [--overwrite]
//!       [--fabric-class C --fabric-owner O --fabric-job J --fabric-card CARD --fabric-cmd STR]
//!       (fabric fields fall back to env FABRIC_CLASS/OWNER/JOB/CARD/CMD and are only recorded in the receipt)
//!   Run only on a card granted by the fabric scheduler, e.g. (class `correct` for synthetic/real, `timing` for timing):
//!   /home/kaden/qcal/locks/fabric/fabric-run.sh --class correct --card A --mem-gb 8 --priority P1 \
//!       --owner Subq4WeekLead3.R8E8pGp --max-secs N -- target/release/examples/e8p_i4_moe_gemv_oracle synthetic ...
//!   Exit code: 0 all exact, 3 ran but something mismatched (receipt still written), 1 error (usage, device, I/O).
//!   `--negative-control` flips one expected bit per launch and must exit 3 (it proves the comparison sees a single bit).
//!
//! # Modes
//! * `synthetic`: 27 cases. Wire: every case mixes the four generators of the wire words (`random`; `every_word`, a per-block
//!   bijective scramble in which each of the 65536 wire words appears in every 65536-word block, so both shift values,
//!   both lane parities and all 128 `sign7` values meet every abs row; `extreme`, the max-`|v|` abs rows with all-negative
//!   and all-positive signs and shift both ways; `parity`, odd and even abs rows with `popcount(sign7)` both parities) and
//!   scale classes `normal`, FP16 max `0x7bff`, guard boundary `0x0400`, low normals `0x0401/0x0403/0x0405` where `d/4` is
//!   not exact in FP16, `0x3c00`, and every class on every row/group in the `*_scale_edges` case (all written scales pass
//!   `writer_guard`). Slots: `x_row_div` 10 (gate) and 1 (down); C1..C8 with all experts distinct (`n = 1`), all lanes sharing
//!   the same ten experts (`n = C`, read-once), partial reuse (`c8_mixed_reuse`, ids 0 and 511 included), planner-capacity
//!   descriptors whose tail experts have no hit (`c2_nohit_tail`), a bucket with no hit at all (`c1_empty_bucket`),
//!   `n > 8`, `end > 10*C` (refused, outputs keep the sentinel, neighbours still computed), `x_src_rows < rows` (rows read `0`),
//!   padded `x_row_stride`/`y_slot_stride` (pads keep poison). Expert blobs are separate allocations, each exactly the wire
//!   size with 256 B of `0xFF` before and after (a scale word read from the guard is NaN; the guards are re-read after
//!   every launch and must be unchanged), `Y` outputs are filled `0xa5` and carry a trailing guard region compared bit
//!   for bit, `X` pads and a trailing guard are NaN. Final-row scales are real scales followed directly by guard poison.
//! * `real`: all 512 solved E8P experts of layer 0 (both projections) from the existing pinned records
//!   (`L0.gate_up_proj.bin` sha b62e40c7.., `L0.down_proj.bin` sha 7ccdc29a.., manifest sha b60856a1..): whole-file and
//!   per-record hashes, manifest offsets/sizes, header, `books_sha256`, writer guard on every scale, a repack whose inverse is
//!   checked through `unpack_word_fields`/`pack_word` (the repack is the one of `hipfire-arch-qwen4/examples/e8p_gp/records.rs`;
//!   no encoder, no solve). Seven C8 cover cases visit every expert id at least once for both projections (coverage count is
//!   asserted to be 512/512), plus a shared case, an edge case and a mixed-reuse case. Activations stay synthetic.
//! * `timing`: same card, same session, the new E8P module vs the frozen MQ4 F4 module (`mq_i4_moe_gemv_pm_gfx1201`,
//!   `mq4sym` R256/R128, same kernarg, same descriptors) at C1..C8, `distinct` (every slot its own expert) and `shared`
//!   (all lanes the same ten experts, read-once) routes, resident (same buffers every launch, `batch` launches per
//!   sample) and L2-cold (a 256 MB device copy evicts L2/MALL before the start event, then `batch` launches of distinct
//!   expert sets, nothing evicts inside the measured interval). Sample = `hipEventElapsedTime` over the batch / launches
//!   (launch-inclusive), warmup 5 / reps 25, codecs interleaved per repetition with alternating order,
//!   `row_us = launch_us / (M * slots)`. The E8P output of every timed route is verified byte exact against the twin
//!   first. No threshold is applied: ratios `e8p/mq4` are reported and nothing is killed. Scope: synthetic MQ4 data (random
//!   codes/scales of the legacy wire) vs real (if records are given) or synthetic E8P data; timing is data independent.
//!
//! # Receipts
//! One JSON (stdout and `--out`): schema, mode, argv, build command, device identity (uuid/bdf/arch/card), fabric fields,
//! sha256 of the harness source, of the K-E0 reference source (pinned `bc55ff77..`), of the launcher source and of the
//! code objects, record file/manifest pins, per case input sha256, actual launch / word / mismatch / guard-region counts,
//! poison checks and `rc`.
//!
//! # gfx1151 (Strix Halo; `--expect-arch gfx1151`)
//! The same oracle against the gfx1151 image: the launcher (`E8pArch::Gfx1151`, chosen from the device) binds module `e8p_i4_moe_gemv_pm_gfx1151` and the
//! symbols `e8p_i4_moe_gemv_{r256_gate_up,r128_down}_pm_gfx1151`; `plan_launch_on` must reproduce the kernarg bytes and name the gfx1151 symbol; the slab
//! scratch path (`Gpu::e8p_i4_moe_gemv_slab`) runs through the same arch. The twin, the cases, the poison / guard checks and the exit codes are unchanged
//! (outputs must be bit-identical to the gfx1201 kernel and the K-E0 twin). Defaults stay gfx1201. gfx1151 has no ROCr UUID: `--expect-uuid` is optional
//! (checked only when given; the device uuid is recorded verbatim, `uuid_authoritative: false`) and `--expect-bdf` is REQUIRED (`lspci -D -d 1002:` on
//! hipx). `timing` is refused on gfx1151 (its baseline is the frozen `mq_i4_moe_gemv_pm_gfx1201` MQ4 module; no gfx1151 MQ4 F4 module exists).
//!   cargo build -p hipfire-runtime --release --features lab --example e8p_i4_moe_gemv_oracle
//!   /home/kaden/qcal/locks/fabric/fabric-run.sh --class correct --card halo --mem-gb N --owner NAME -- \
//!     target/release/examples/e8p_i4_moe_gemv_oracle synthetic --e8p-codeobject /abs/e8p_i4_moe_gemv_pm_gfx1151.hxaco \
//!     --out /abs/r10.k1.e8p-gfx1151.synthetic.json --expect-arch gfx1151 --expect-bdf 0000:XX:00.0 [--expect-uuid U] --card halo [--overwrite]
//!   `real` (all 512 solved layer-0 vq-e8p records, arch independent) takes the same flags plus `--records-dir DIR --manifest FILE` where those pinned
//!   records are on the card's host. The real layers of the shipped Halo slab artifact, layers 0 and 47 (four cold experts, both projections) of
//!   hipx `/home/kaden/pm-wave/consumer24/models/c24h6306-d2/qwen3.8-flash-next.mq2-xts`, are covered by the K1 G1 harness
//!   `e8p_i4_moe_gemv_gfx1100_g1 --expect-arch gfx1151` (its header has the extraction and run commands).

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;
use std::sync::Arc;

use hip_bridge::{DeviceBuffer, Function, HipRuntime};
use hipfire_runtime::e8p_i4_reference as e8p;
use hipfire_runtime::mq_i4::MqI4Rotation;
use hipfire_runtime::mq_i4_reference::gemv_tree_f32_q;
use rayon::prelude::*;
use rdna_compute::e8p_moe_decode::{
    max_union, plan_launch_on, union_bucket_words, union_words, DevView, E8pArch, E8pDecodeRequest,
    E8pKernargs, E8pKind, E8pMoeDecodeModule, E8pUnionHost, E8P_I4_MOE_GEMV_SYMBOLS, KERNARG_BYTES,
    MAX_EXPERTS, MAX_LANES, MAX_SLOTS, TOP_K,
};
use rdna_compute::{DType, Gpu};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

type Result<T, E = String> = std::result::Result<T, E>;
fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

const SCHEMA: &str = "e8p-i4-moe-gemv-oracle-v1";
const BUILD_COMMAND: &str =
    "cargo build -p hipfire-runtime --release --features lab --example e8p_i4_moe_gemv_oracle";
const HARNESS_SRC: &[u8] = include_bytes!("e8p_i4_moe_gemv_oracle.rs");
const REFERENCE_SRC: &[u8] = include_bytes!("../src/e8p_i4_reference.rs");
const LAUNCHER_SRC: &[u8] = include_bytes!("../../rdna-compute/src/e8p_moe_decode.rs");
/// Frozen K-E0 reference source (`runtime/e8p_i4_reference.rs`). Re-pinned from `17b9e138..` after the rc5 land
/// rustfmt'd it and moved two test-only fixture env reads to `hipfire_config::developer_var` (no non-test change).
const REFERENCE_SHA256: &str = "bc55ff77a110a679ea0e71b1b0b52144a653384821afe4730840304ea5bbe130";
const MANIFEST_SHA256: &str = "b60856a1260919c63bdf47523053399aeaca04f1e75fbda9a4efb7020add4611";
const L0_GATE_SHA256: &str = "b62e40c7119b8aee26c6204937bcd80666f3705aab58fe1702ca336fcef14dfc";
const L0_DOWN_SHA256: &str = "7ccdc29ae35fb2775ac292c49ebcc0e7e5ac9db40c344d5dd0aa70ecfb643bac";
const MQ4_SYMBOLS: [&str; 2] = [
    "mq_i4_moe_gemv_mq4sym_r256_gate_up",
    "mq_i4_moe_gemv_mq4sym_r128_down",
];
const RECORD_HEADER: [u8; 8] = *b"e8p\0\0\0\0\0";

const SENTINEL_BITS: u32 = 0xa5a5_a5a5;
const NAN_BITS: u32 = 0x7fc0_0000;
const GUARD: usize = 256;
/// Trailing guard words after every `Y` / `X` buffer.
const TAIL_WORDS: usize = 256;
const FIRST_LIMIT: usize = 8;
const N_EXPERTS: usize = MAX_EXPERTS;

fn sha_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn bits_hex(v: u32) -> String {
    format!("{v:#010x}")
}
fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn u32_bytes(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn u64_bytes(v: &[u64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn rotation(kind: E8pKind) -> MqI4Rotation {
    match kind {
        E8pKind::GateUp => MqI4Rotation::R256,
        E8pKind::Down => MqI4Rotation::R128,
    }
}
fn kind_name(kind: E8pKind) -> &'static str {
    match kind {
        E8pKind::GateUp => "gate_up",
        E8pKind::Down => "down",
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Rng (splitmix; same family as the K1 oracle)
// ---------------------------------------------------------------------------------------------------------------

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
    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = self.below(i + 1);
            v.swap(i, j);
        }
    }
}
fn name_seed(name: &str, seed: u64) -> u64 {
    let d = Sha256::digest(name.as_bytes());
    u64::from_le_bytes(d[..8].try_into().unwrap()) ^ seed
}

// ---------------------------------------------------------------------------------------------------------------
// Synthetic wire: experts
// ---------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum WordMode {
    Random,
    EveryWord,
    Extreme,
    Parity,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ScaleMode {
    Normal,
    Edge,
}

fn word_mode_of(id: u32) -> WordMode {
    match id % 4 {
        0 => WordMode::Random,
        1 => WordMode::EveryWord,
        2 => WordMode::Extreme,
        _ => WordMode::Parity,
    }
}
fn scale_mode_of(id: u32) -> ScaleMode {
    if id % 3 == 2 {
        ScaleMode::Edge
    } else {
        ScaleMode::Normal
    }
}

/// FP16 scale bits of a row/group class (all pass the writer guard).
fn scale_bits(sm: ScaleMode, r: usize, g: usize, salt: usize, rng: &mut Rng) -> u16 {
    let class = match sm {
        ScaleMode::Normal => 0,
        ScaleMode::Edge => (r * 5 + g * 3 + salt) % 10,
    };
    let bits = match class {
        0 => (((8 + rng.below(10)) as u16) << 10) | (rng.u32() as u16 & 0x3ff),
        1 => 0x7bff,
        2 => 0x0400,
        3 => 0x0401,
        4 => 0x0403,
        5 => 0x0405,
        6 => 0x3c00,
        7 => 0x0400 + rng.below(0x100) as u16,
        8 => 0x6000 + rng.below(0x1bff) as u16,
        _ => 0x7bfe,
    };
    debug_assert!(e8p::writer_guard(bits).is_ok());
    bits
}

/// Planar wire blob `[words: M*K/8 u16 LE][scales: M*K/128 FP16 LE]` of one expert projection.
fn gen_blob(kind: E8pKind, id: u32, wm: WordMode, sm: ScaleMode) -> Vec<u8> {
    let (m, k) = (kind.m(), kind.k());
    let (nw, ns, groups) = (m * k / 8, m * k / 128, k / 128);
    let mut rng = Rng::new(mix64(
        ((id as u64) << 16) | ((kind.index() as u64) << 8) | wm as u64 * 4 + sm as u64,
    ));
    // Extreme means largest decoded component, not largest row norm: norm²48
    // rows contain only 1s and 3s, whereas the largest component is 5.
    let max_abs2 = *e8p::E8P_ABS2.iter().flatten().max().unwrap();
    let hi: Vec<u8> = (0..=255u8)
        .filter(|&i| e8p::E8P_ABS2[i as usize].contains(&max_abs2))
        .collect();
    let mut out = Vec::with_capacity(kind.blob_bytes());
    for i in 0..nw {
        let w: u16 = match wm {
            WordMode::Random => rng.u32() as u16,
            WordMode::EveryWord => {
                let (j, b) = ((i % 65536) as u32, (i / 65536) as u32 + id);
                ((j.wrapping_mul(40503)).wrapping_add(b.wrapping_mul(977)) & 0xffff) as u16
            }
            WordMode::Extreme => {
                let idx = hi[i % hi.len()];
                let sign7 = if (i / hi.len()) % 2 == 0 { 0 } else { 127 };
                e8p::pack_word(idx, sign7, (i / (2 * hi.len())) % 2 == 1)
            }
            WordMode::Parity => {
                let idx = ((i * 7 + id as usize) % 256) as u8;
                let sign7 = ((i * 37 + (i >> 8)) & 0x7f) as u8;
                e8p::pack_word(idx, sign7, i & 1 == 1)
            }
        };
        out.extend_from_slice(&w.to_le_bytes());
    }
    for s in 0..ns {
        let d = scale_bits(sm, s / groups, s % groups, id as usize, &mut rng);
        out.extend_from_slice(&d.to_le_bytes());
    }
    assert_eq!(out.len(), kind.blob_bytes());
    out
}

struct ExpertBlobs {
    gate: Vec<u8>,
    down: Vec<u8>,
    gate_sha: String,
    down_sha: String,
}
impl ExpertBlobs {
    fn new(gate: Vec<u8>, down: Vec<u8>) -> Self {
        let (gate_sha, down_sha) = (sha_hex(&gate), sha_hex(&down));
        ExpertBlobs {
            gate,
            down,
            gate_sha,
            down_sha,
        }
    }
    fn proj(&self, kind: E8pKind) -> &[u8] {
        match kind {
            E8pKind::GateUp => &self.gate,
            E8pKind::Down => &self.down,
        }
    }
}
type Experts = BTreeMap<u32, Arc<ExpertBlobs>>;
type ExpertCache = HashMap<(u32, WordMode, ScaleMode), Arc<ExpertBlobs>>;

fn experts_for(
    cache: &mut ExpertCache,
    ids: &[u32],
    wm: &dyn Fn(u32) -> WordMode,
    sm: &dyn Fn(u32) -> ScaleMode,
) -> Experts {
    let mut missing: Vec<(u32, WordMode, ScaleMode)> = Vec::new();
    for &id in ids {
        let key = (id, wm(id), sm(id));
        if !cache.contains_key(&key) && !missing.contains(&key) {
            missing.push(key);
        }
    }
    let made: Vec<((u32, WordMode, ScaleMode), Arc<ExpertBlobs>)> = missing
        .par_iter()
        .map(|&key| {
            let (id, w, s) = key;
            let e = ExpertBlobs::new(
                gen_blob(E8pKind::GateUp, id, w, s),
                gen_blob(E8pKind::Down, id, w, s),
            );
            (key, Arc::new(e))
        })
        .collect();
    cache.extend(made);
    ids.iter()
        .map(|&id| (id, cache[&(id, wm(id), sm(id))].clone()))
        .collect()
}

fn gen_x_row(style: usize, n: usize, rng: &mut Rng) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let v = match style % 4 {
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
            _ => [0.0f32, -0.0, 1.0e-30, -1.0e-30, 1.0, -1.0, 0.5, -0.5][i % 8],
        };
        out.push(v);
    }
    out
}

fn pick_ids(rng: &mut Rng, n: usize, edges: bool) -> Vec<u32> {
    let mut set = BTreeSet::new();
    if edges {
        set.insert(0u32);
        set.insert(511u32);
    }
    while set.len() < n {
        set.insert(rng.below(N_EXPERTS) as u32);
    }
    set.into_iter().collect()
}

// ---------------------------------------------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------------------------------------------

/// One bucket descriptor exactly as the kernel reads it (possibly out of contract on purpose).
#[derive(Clone, Default)]
struct Desc {
    ids: Vec<u32>,
    offs: Vec<u32>,
    slots: Vec<u32>,
}

#[derive(Clone, Copy)]
struct Geom {
    /// F32 elements between X rows.
    xrs: usize,
    /// F32 elements between original-slot output rows.
    ys: usize,
    /// kernarg `x_src_rows`.
    x_src_rows: usize,
}

struct Case {
    name: String,
    origin: &'static str,
    c: usize,
    desc: Desc,
    experts: Experts,
    /// Allocated rows `x_src_rows(kind, c)` of `xrs` words each (pads NaN).
    x: [Vec<f32>; 2],
    geom: [Geom; 2],
    /// Frozen geometry (tight strides, full `x_src_rows`): the launcher's `plan_launch` must reproduce the kernarg.
    tight: bool,
}

impl Case {
    fn slots(&self) -> usize {
        TOP_K * self.c
    }
    fn n_words(&self, kind: E8pKind) -> usize {
        self.slots() * self.geom[kind.index()].ys + TAIL_WORDS
    }
    /// `true` when the descriptor is a legal planner output (so the slab scratch path may replay it).
    fn desc_in_contract(&self) -> bool {
        let s = self.slots();
        let d = &self.desc;
        let u = d.ids.len();
        if u == 0 || u > max_union(self.c) || d.offs.len() != u + 1 || d.offs[0] != 0 {
            return false;
        }
        if !d
            .offs
            .windows(2)
            .all(|w| w[0] <= w[1] && w[1] - w[0] <= MAX_SLOTS as u32)
        {
            return false;
        }
        let end = d.offs[u] as usize;
        end <= s
            && d.slots.len() >= end
            && d.slots[..end].iter().all(|&x| (x as usize) < s)
            && d.ids.iter().all(|&e| (e as usize) < N_EXPERTS)
    }
    fn input_sha256(&self) -> String {
        let mut h = Sha256::new();
        h.update(self.name.as_bytes());
        h.update((self.c as u64).to_le_bytes());
        h.update(u32_bytes(&self.desc.ids));
        h.update(u32_bytes(&self.desc.offs));
        h.update(u32_bytes(&self.desc.slots));
        for g in &self.geom {
            for v in [g.xrs, g.ys, g.x_src_rows] {
                h.update((v as u64).to_le_bytes());
            }
        }
        for (id, e) in &self.experts {
            h.update(id.to_le_bytes());
            h.update(e.gate_sha.as_bytes());
            h.update(e.down_sha.as_bytes());
        }
        h.update(f32_bytes(&self.x[0]));
        h.update(f32_bytes(&self.x[1]));
        h.update([self.tight as u8]);
        sha_hex(&h.finalize())
    }
}

#[derive(Clone, Copy, Default)]
struct CaseOpts {
    padded: bool,
    x_oob: bool,
}

#[allow(clippy::too_many_arguments)]
fn build_case(
    name: &str,
    origin: &'static str,
    c: usize,
    route: &[u32],
    desc: Option<Desc>,
    experts: Experts,
    opts: CaseOpts,
    seed: u64,
) -> Result<Case> {
    if route.len() != TOP_K * c {
        return Err(format!(
            "{name}: route has {} entries, expected {}",
            route.len(),
            TOP_K * c
        ));
    }
    let desc = match desc {
        Some(d) => d,
        None => {
            let u = E8pUnionHost::from_route(route, |_| true).map_err(err)?;
            Desc {
                ids: u.expert_ids,
                offs: u.expert_offsets,
                slots: u.slot_ids,
            }
        }
    };
    let mut rng = Rng::new(name_seed(name, seed));
    let mut geom = [Geom {
        xrs: 0,
        ys: 0,
        x_src_rows: 0,
    }; 2];
    let mut x: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
    for kind in E8pKind::ALL {
        let rows = kind.x_src_rows(c);
        let (xrs, ys) = if opts.padded {
            (kind.k() + 64, kind.y_slot_stride() + 32)
        } else {
            (kind.k(), kind.y_slot_stride())
        };
        // Rows past `x_src_rows` read zero: the last lane's gate row / the last lane's ten down rows.
        let x_src_rows = if opts.x_oob {
            rows - if kind == E8pKind::GateUp { 1 } else { TOP_K }
        } else {
            rows
        };
        geom[kind.index()] = Geom {
            xrs,
            ys,
            x_src_rows,
        };
        let mut buf = vec![f32::from_bits(NAN_BITS); rows * xrs + TAIL_WORDS];
        for r in 0..rows {
            let row = gen_x_row(r + kind.index(), kind.k(), &mut rng);
            buf[r * xrs..r * xrs + kind.k()].copy_from_slice(&row);
        }
        x[kind.index()] = buf;
    }
    Ok(Case {
        name: name.to_string(),
        origin,
        c,
        desc,
        experts,
        x,
        geom,
        tight: !opts.padded && !opts.x_oob,
    })
}

fn lanes_to_route(lanes: &[Vec<u32>]) -> Vec<u32> {
    lanes.iter().flatten().copied().collect()
}

/// Expert ids the kernel must reject without a table load or a store: `-1`, the table extent, `INT_MAX`, `INT_MIN`.
const BAD_IDS: [u32; 4] = [u32::MAX, N_EXPERTS as u32, i32::MAX as u32, i32::MIN as u32];

const SYNTH_CASES: &[&str] = &[
    "c1_distinct",
    "c2_distinct",
    "c3_distinct",
    "c4_distinct",
    "c5_distinct",
    "c6_distinct",
    "c7_distinct",
    "c8_distinct",
    "c2_shared",
    "c3_shared",
    "c4_shared",
    "c5_shared",
    "c6_shared",
    "c7_shared",
    "c8_shared",
    "c8_mixed_reuse",
    "c5_edge_ids",
    "c3_padded",
    "c4_x_oob",
    "c2_nohit_tail",
    "c1_empty_bucket",
    "c1_refused_n9_and_end",
    "c1_bad_ids",
    "c2_wire_random",
    "c2_wire_everyword",
    "c2_wire_extreme",
    "c2_wire_parity",
    "c2_scale_edges",
];

fn build_synthetic(name: &str, cache: &mut ExpertCache, seed: u64) -> Result<Case> {
    let mut rng = Rng::new(name_seed(name, seed) ^ 0x5e17);
    let c: usize = name
        .strip_prefix('c')
        .and_then(|s| s.split('_').next())
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| format!("{name}: no lane count"))?;
    let mixed = (
        &word_mode_of as &dyn Fn(u32) -> WordMode,
        &scale_mode_of as &dyn Fn(u32) -> ScaleMode,
    );
    let distinct_route = |rng: &mut Rng, edges: bool| -> Vec<u32> {
        let mut ids = pick_ids(rng, TOP_K * c, edges);
        rng.shuffle(&mut ids);
        ids
    };
    let shared_route = |rng: &mut Rng| -> Vec<u32> {
        let base = pick_ids(rng, TOP_K, false);
        let mut route = Vec::new();
        for _ in 0..c {
            let mut lane = base.clone();
            rng.shuffle(&mut lane);
            route.extend(lane);
        }
        route
    };
    let build = |cache: &mut ExpertCache,
                 route: &[u32],
                 wm: &dyn Fn(u32) -> WordMode,
                 sm: &dyn Fn(u32) -> ScaleMode,
                 desc: Option<Desc>,
                 opts: CaseOpts| {
        let experts = experts_for(cache, route, wm, sm);
        build_case(name, "synthetic", c, route, desc, experts, opts, seed)
    };
    match name {
        n if n.ends_with("_distinct") => {
            let route = distinct_route(&mut rng, c >= 2);
            build(cache, &route, mixed.0, mixed.1, None, CaseOpts::default())
        }
        n if n.ends_with("_shared") => {
            let route = shared_route(&mut rng);
            build(cache, &route, mixed.0, mixed.1, None, CaseOpts::default())
        }
        "c8_mixed_reuse" => {
            let pool = pick_ids(&mut rng, 24, true);
            let mut route = Vec::new();
            for _ in 0..c {
                let mut p = pool.clone();
                rng.shuffle(&mut p);
                route.extend_from_slice(&p[..TOP_K]);
            }
            build(cache, &route, mixed.0, mixed.1, None, CaseOpts::default())
        }
        "c5_edge_ids" => {
            let route = distinct_route(&mut rng, true);
            build(cache, &route, mixed.0, mixed.1, None, CaseOpts::default())
        }
        "c3_padded" => {
            let route = distinct_route(&mut rng, false);
            build(
                cache,
                &route,
                mixed.0,
                mixed.1,
                None,
                CaseOpts {
                    padded: true,
                    x_oob: false,
                },
            )
        }
        "c4_x_oob" => {
            let route = distinct_route(&mut rng, false);
            build(
                cache,
                &route,
                mixed.0,
                mixed.1,
                None,
                CaseOpts {
                    padded: false,
                    x_oob: true,
                },
            )
        }
        "c2_nohit_tail" => {
            // Planner-capacity descriptor: S = 10 C entries, the tail experts have no slot (n = 0).
            let route = distinct_route(&mut rng, false);
            let u = E8pUnionHost::from_route(&route, |_| true).map_err(err)?;
            let s = TOP_K * c;
            let n = *u.expert_offsets.last().unwrap();
            let mut ids = u.expert_ids.clone();
            ids.resize(s, 0);
            let mut offs = u.expert_offsets.clone();
            offs.resize(s + 1, n);
            build(
                cache,
                &route,
                mixed.0,
                mixed.1,
                Some(Desc {
                    ids,
                    offs,
                    slots: u.slot_ids,
                }),
                CaseOpts::default(),
            )
        }
        "c1_empty_bucket" => {
            // Nothing hits this bucket: every expert has n = 0 and no output word may change.
            let route = distinct_route(&mut rng, false);
            let desc = Desc {
                ids: route.clone(),
                offs: vec![0; route.len() + 1],
                slots: vec![0xffff_ffff; route.len()],
            };
            build(
                cache,
                &route,
                mixed.0,
                mixed.1,
                Some(desc),
                CaseOpts::default(),
            )
        }
        "c1_refused_n9_and_end" => {
            // E0: n = 9 > 8 (refused); E1: slot 9 (computed); E2: n = 7 but end 17 > 10 C (refused, slot_ids never read).
            let route = distinct_route(&mut rng, false);
            let desc = Desc {
                ids: vec![route[0], route[9], route[5]],
                offs: vec![0, 9, 10, 17],
                slots: (0..10).collect(),
            };
            build(
                cache,
                &route,
                mixed.0,
                mixed.1,
                Some(desc),
                CaseOpts::default(),
            )
        }
        "c1_bad_ids" => {
            // Signed-bad expert ids at the table-extent boundary (-1, n_experts, INT_MAX, INT_MIN) interleaved with two valid
            // experts, one slot each: the kernel's one unsigned compare exits before the table is indexed and nothing is
            // stored for the bad ones (their slots keep the canary), the valid neighbours are computed.
            let route = distinct_route(&mut rng, false);
            let desc = Desc {
                ids: vec![
                    route[0], BAD_IDS[0], BAD_IDS[1], BAD_IDS[2], BAD_IDS[3], route[9],
                ],
                offs: vec![0, 1, 2, 3, 4, 5, 6],
                slots: vec![0, 1, 2, 3, 4, 9],
            };
            build(
                cache,
                &route,
                mixed.0,
                mixed.1,
                Some(desc),
                CaseOpts::default(),
            )
        }
        "c2_wire_random" | "c2_wire_everyword" | "c2_wire_extreme" | "c2_wire_parity"
        | "c2_scale_edges" => {
            let wm = match name {
                "c2_wire_everyword" => WordMode::EveryWord,
                "c2_wire_extreme" => WordMode::Extreme,
                "c2_wire_parity" => WordMode::Parity,
                _ => WordMode::Random,
            };
            let sm = if name == "c2_scale_edges" {
                ScaleMode::Edge
            } else {
                ScaleMode::Normal
            };
            let route = shared_route(&mut rng);
            build(
                cache,
                &route,
                &move |_| wm,
                &move |_| sm,
                None,
                CaseOpts::default(),
            )
        }
        _ => Err(format!("unknown synthetic case {name}")),
    }
}

// ---------------------------------------------------------------------------------------------------------------
// CPU expected buffers: the frozen K-E0 lane-pair twin
// ---------------------------------------------------------------------------------------------------------------

struct Expected {
    y0: Vec<f32>,
    y1: Option<Vec<f32>>,
    selfcheck_bad: u64,
    twin_rows: u64,
    experts: BTreeSet<u32>,
    written_slots: usize,
}

fn parse_blob(kind: E8pKind, blob: &[u8]) -> (Vec<u16>, Vec<u16>) {
    let (nw, ns) = (kind.m() * kind.k() / 8, kind.m() * kind.k() / 128);
    let words = blob[..nw * 2]
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    let scales = blob[nw * 2..nw * 2 + ns * 2]
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    (words, scales)
}

fn expected(case: &Case, kind: E8pKind, negative: bool) -> Result<Expected> {
    let g = case.geom[kind.index()];
    let x = &case.x[kind.index()];
    let (m, k, out) = (kind.m(), kind.k(), kind.out_rows());
    let slots = case.slots();
    let n_words = case.n_words(kind);
    let sentinel = f32::from_bits(SENTINEL_BITS);
    let mut y0 = vec![sentinel; n_words];
    let mut y1 = kind.has_y1().then(|| vec![sentinel; n_words]);
    // The kernel's validation, replayed: n = 0, n > 8 or end > 10 C returns without a store.
    let d = &case.desc;
    let mut work: Vec<(u32, Vec<u32>)> = Vec::new();
    for u in 0..d.ids.len() {
        let (lo, hi) = (d.offs[u], d.offs[u + 1]);
        let n = hi.wrapping_sub(lo);
        // An id at or past the table extent (unsigned, so a negative id too) exits before the table is indexed.
        if n == 0 || n > MAX_SLOTS as u32 || hi as usize > slots || d.ids[u] >= N_EXPERTS as u32 {
            continue;
        }
        work.push((d.ids[u], d.slots[lo as usize..hi as usize].to_vec()));
    }
    let zeros = vec![0f32; k];
    let rot = rotation(kind);
    let results: Vec<(Vec<Vec<f32>>, u64)> = work
        .par_iter()
        .enumerate()
        .map(|(wi, (id, sl))| -> Result<(Vec<Vec<f32>>, u64)> {
            let blob = case
                .experts
                .get(id)
                .ok_or_else(|| format!("{}: expert {id} has no blob", case.name))?
                .proj(kind);
            let (words, scales) = parse_blob(kind, blob);
            let q = e8p::decode_weights_q(m, k, &words);
            let dq: Vec<f32> = scales.iter().map(|&b| e8p::scale_quarter_f32(b)).collect();
            let mut rows = Vec::with_capacity(sl.len());
            let mut bad = 0u64;
            for (si, &slot) in sl.iter().enumerate() {
                if slot as usize >= slots {
                    return Err(format!("{}: slot {slot} outside 10 C", case.name));
                }
                let r = slot as usize / kind.x_row_div();
                let xs: &[f32] = if r >= g.x_src_rows {
                    &zeros
                } else {
                    &x[r * g.xrs..r * g.xrs + k]
                };
                let y = gemv_tree_f32_q(rot, m, k, &q, &dq, xs);
                if wi == 0 && si == 0 {
                    // The lane-pair twin itself, end to end, for the first slot of the launch.
                    let t = e8p::decode_gemv_f32_lanepair(rot, m, k, &words, &scales, xs);
                    bad += y
                        .iter()
                        .zip(&t)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count() as u64;
                }
                rows.push(y);
            }
            Ok((rows, bad))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut selfcheck_bad = 0u64;
    let mut twin_rows = 0u64;
    let mut experts = BTreeSet::new();
    let mut written_slots = 0usize;
    for ((id, sl), (rows, bad)) in work.iter().zip(&results) {
        selfcheck_bad += bad;
        experts.insert(*id);
        for (&slot, y) in sl.iter().zip(rows) {
            written_slots += 1;
            twin_rows += y.len() as u64;
            for (r, &v) in y.iter().enumerate() {
                if r < out {
                    y0[slot as usize * g.ys + r] = v;
                } else {
                    y1.as_mut().expect("gate/up")[slot as usize * g.ys + r - out] = v;
                }
            }
        }
    }
    if negative {
        if let Some((_, sl)) = work.first() {
            let at = sl[0] as usize * g.ys;
            y0[at] = f32::from_bits(y0[at].to_bits() ^ 1);
        }
    }
    Ok(Expected {
        y0,
        y1,
        selfcheck_bad,
        twin_rows,
        experts,
        written_slots,
    })
}

// ---------------------------------------------------------------------------------------------------------------
// Device plumbing
// ---------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Buf {
    idx: usize,
    ptr: u64,
    len: usize,
}
#[derive(Default)]
struct Dev {
    bufs: Vec<DeviceBuffer>,
}
impl Dev {
    fn alloc(&mut self, hip: &HipRuntime, n: usize) -> Result<Buf> {
        let b = hip.malloc(n.max(4)).map_err(err)?;
        let ptr = b.as_ptr() as usize as u64;
        self.bufs.push(b);
        Ok(Buf {
            idx: self.bufs.len() - 1,
            ptr,
            len: n,
        })
    }
    fn upload(&mut self, hip: &HipRuntime, bytes: &[u8]) -> Result<Buf> {
        let b = self.alloc(hip, bytes.len())?;
        hip.memcpy_htod(&self.bufs[b.idx], bytes).map_err(err)?;
        Ok(b)
    }
    fn fill(&mut self, hip: &HipRuntime, n: usize, value: i32) -> Result<Buf> {
        let b = self.alloc(hip, n)?;
        hip.memset(&self.bufs[b.idx], value, n.max(4))
            .map_err(err)?;
        Ok(b)
    }
    fn download(&self, hip: &HipRuntime, b: Buf, n: usize) -> Result<Vec<u8>> {
        let mut out = vec![0u8; n];
        hip.memcpy_dtoh(&mut out, &self.bufs[b.idx]).map_err(err)?;
        Ok(out)
    }
    fn release(self, hip: &HipRuntime) {
        for b in self.bufs {
            let _ = hip.free(b);
        }
    }
}

/// A weight blob: its own allocation, `GUARD` bytes of `0xFF` before and after, the blob exactly `bytes.len()` long.
struct BlobAlloc {
    buf: Buf,
    base: Buf,
    len: usize,
}
fn upload_blob(dev: &mut Dev, hip: &HipRuntime, bytes: &[u8]) -> Result<BlobAlloc> {
    let mut v = vec![0xffu8; GUARD + bytes.len() + GUARD];
    v[GUARD..GUARD + bytes.len()].copy_from_slice(bytes);
    let base = dev.upload(hip, &v)?;
    Ok(BlobAlloc {
        buf: Buf {
            idx: base.idx,
            ptr: base.ptr + GUARD as u64,
            len: bytes.len(),
        },
        base,
        len: bytes.len(),
    })
}
/// Re-reads both guards; returns `(regions_checked, regions_changed)`.
fn check_guards(dev: &Dev, hip: &HipRuntime, blobs: &[BlobAlloc]) -> Result<(u64, u64)> {
    let (mut checked, mut changed) = (0u64, 0u64);
    for b in blobs {
        for off in [0usize, GUARD + b.len] {
            let mut g = vec![0u8; GUARD];
            hip.memcpy_dtoh_at(&mut g, &dev.bufs[b.base.idx], off)
                .map_err(err)?;
            checked += 1;
            changed += g.iter().any(|&x| x != 0xff) as u64;
        }
    }
    Ok((checked, changed))
}

fn kernargs_of(kind: E8pKind, c: usize, u: usize, ptrs: [u64; 7], g: Geom) -> E8pKernargs {
    E8pKernargs {
        expert_weight_ptrs: ptrs[0],
        expert_ids: ptrs[1],
        expert_offsets: ptrs[2],
        slot_ids: ptrs[3],
        xrot: ptrs[4],
        y0: ptrs[5],
        y1: ptrs[6],
        m: kind.m() as u32,
        k: kind.k() as u32,
        u: u as u32,
        c: c as u32,
        x_row_div: kind.x_row_div() as u32,
        x_src_rows: g.x_src_rows as u32,
        x_row_stride: g.xrs as u32,
        y_slot_stride: g.ys as u32,
        // The pointer table every launch of this oracle uploads has exactly `N_EXPERTS` entries.
        n_experts: N_EXPERTS as u32,
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Context, totals, comparison
// ---------------------------------------------------------------------------------------------------------------

struct Ctx {
    gpu: Gpu,
    module: E8pMoeDecodeModule,
    image: Arc<[u8]>,
    negative: bool,
    slab_ready: bool,
}

#[derive(Default)]
struct Totals {
    cases: u64,
    kind_runs: u64,
    launches: u64,
    mismatches: u64,
    selfcheck_bad: u64,
    twin_rows: u64,
    plan_checks: u64,
    plan_mismatches: u64,
    slab_runs: u64,
    slab_mismatches: u64,
    guard_regions: u64,
    guard_changed: u64,
    words_compared: u64,
    pad_words_compared: u64,
    experts_gate: BTreeSet<u32>,
    experts_down: BTreeSet<u32>,
}

#[derive(Default)]
struct Mm {
    count: u64,
    first: Vec<Value>,
}
impl Mm {
    fn push(&mut self, v: Value) {
        self.count += 1;
        if self.first.len() < FIRST_LIMIT {
            self.first.push(v);
        }
    }
}

fn compare(
    part: &str,
    gpu: &[u8],
    exp: &[f32],
    ys: usize,
    out_rows: usize,
    slots: usize,
    mm: &mut Mm,
    pad_words: &mut u64,
) {
    for (i, (b, e)) in gpu.chunks_exact(4).zip(exp).enumerate() {
        if i >= slots * ys || i % ys >= out_rows {
            *pad_words += 1;
        }
        let g = u32::from_le_bytes(b.try_into().unwrap());
        if g == e.to_bits() {
            continue;
        }
        let (slot, row) = if i < slots * ys {
            (json!(i / ys), json!(i % ys))
        } else {
            (json!("tail"), json!(i - slots * ys))
        };
        mm.push(json!({"part": part, "word_index": i, "slot": slot, "row": row,
            "gpu_bits": bits_hex(g), "oracle_bits": bits_hex(e.to_bits()), "oracle_unwritten": e.to_bits() == SENTINEL_BITS}));
    }
}

// ---------------------------------------------------------------------------------------------------------------
// One (case, kind) run
// ---------------------------------------------------------------------------------------------------------------

/// Uploads `slot_ids` padded to `10 C` words (poison `0xffffffff`: a stray read would write far out of range).
fn desc_slots_padded(case: &Case) -> Vec<u32> {
    let mut s = case.desc.slots.clone();
    if s.len() < case.slots() {
        s.resize(case.slots(), 0xffff_ffff);
    }
    s
}

fn run_kind(
    ctx: &mut Ctx,
    case: &Case,
    kind: E8pKind,
    slabs: &BTreeMap<u32, (u64, u64)>,
    dev: &mut Dev,
    t: &mut Totals,
) -> Result<Value> {
    let c = case.c;
    let g = case.geom[kind.index()];
    let slots = case.slots();
    let n_words = case.n_words(kind);
    let exp = expected(case, kind, ctx.negative)?;
    t.selfcheck_bad += exp.selfcheck_bad;
    t.twin_rows += exp.twin_rows;
    let set = if kind == E8pKind::GateUp {
        &mut t.experts_gate
    } else {
        &mut t.experts_down
    };
    set.extend(exp.experts.iter().copied());

    let hip = &ctx.gpu.hip;
    let mut table = vec![0u64; N_EXPERTS];
    for (&id, &(gp, dp)) in slabs {
        table[id as usize] = if kind == E8pKind::GateUp { gp } else { dp };
    }
    let tb = dev.upload(hip, &u64_bytes(&table))?;
    let xb = dev.upload(hip, &f32_bytes(&case.x[kind.index()]))?;
    let ib = dev.upload(hip, &u32_bytes(&case.desc.ids))?;
    let ob = dev.upload(hip, &u32_bytes(&case.desc.offs))?;
    let slot_words = desc_slots_padded(case);
    let sb = dev.upload(hip, &u32_bytes(&slot_words))?;
    let y0 = dev.fill(hip, n_words * 4, 0xa5)?;
    let y1 = if kind.has_y1() {
        Some(dev.fill(hip, n_words * 4, 0xa5)?)
    } else {
        None
    };
    let u = case.desc.ids.len();
    let ka = kernargs_of(
        kind,
        c,
        u,
        [
            tb.ptr,
            ib.ptr,
            ob.ptr,
            sb.ptr,
            xb.ptr,
            y0.ptr,
            y1.map_or(0, |b| b.ptr),
        ],
        g,
    );

    // Launcher path: the validated request must reproduce the kernarg bytes exactly on tight geometry.
    let mut plan_checked = false;
    let launcher_path = case.tight && u >= 1 && u <= max_union(c);
    if launcher_path {
        let req = E8pDecodeRequest {
            kind,
            c,
            u,
            expert_ids: DevView::new(ib.ptr, case.desc.ids.len() * 4),
            expert_offsets: DevView::new(ob.ptr, case.desc.offs.len() * 4),
            slot_ids: DevView::new(sb.ptr, slot_words.len() * 4),
            weight_ptrs: DevView::new(tb.ptr, N_EXPERTS * 8),
            n_experts: N_EXPERTS,
            xrot: DevView::new(xb.ptr, case.x[kind.index()].len() * 4),
            y0: DevView::new(y0.ptr, n_words * 4),
            y1: y1.map(|b| DevView::new(b.ptr, n_words * 4)),
        };
        let planned = plan_launch_on(ctx.module.arch(), &req)
            .map_err(|e| format!("{}: plan_launch: {e}", case.name))?;
        t.plan_checks += 1;
        plan_checked = true;
        if planned.kernargs.to_bytes() != ka.to_bytes()
            || planned.grid != [kind.m() as u32, u as u32, 1]
            || planned.symbol != kind.symbol_on(ctx.module.arch())
        {
            t.plan_mismatches += 1;
        }
        // SAFETY: every operand is a live allocation of `dev` of the declared extent, immutable during the launch;
        // `ctx.module` outlives the device synchronize below.
        unsafe { ctx.module.launch(&mut ctx.gpu, &req) }.map_err(err)?;
    } else {
        // SAFETY: as above (explicit out-of-contract strides: only the kernel's own bounds apply).
        unsafe {
            ctx.module
                .launch_kernargs(&mut ctx.gpu, kind, [kind.m() as u32, u as u32, 1], &ka)
        }
        .map_err(err)?;
    }
    t.launches += 1;
    ctx.gpu.hip.device_synchronize().map_err(err)?;

    let hip = &ctx.gpu.hip;
    let mut mm = Mm::default();
    let mut pad = 0u64;
    let g0 = dev.download(hip, y0, n_words * 4)?;
    compare(
        "y0",
        &g0,
        &exp.y0,
        g.ys,
        kind.out_rows(),
        slots,
        &mut mm,
        &mut pad,
    );
    if let (Some(yb), Some(e1)) = (y1, exp.y1.as_ref()) {
        let g1 = dev.download(hip, yb, n_words * 4)?;
        compare(
            "y1",
            &g1,
            e1,
            g.ys,
            kind.out_rows(),
            slots,
            &mut mm,
            &mut pad,
        );
    }
    t.words_compared += (n_words * if kind.has_y1() { 2 } else { 1 }) as u64;
    t.pad_words_compared += pad;
    t.mismatches += mm.count;
    t.kind_runs += 1;

    let mut row = json!({
        "kind": kind_name(kind), "symbol": kind.symbol_on(ctx.module.arch()), "m": kind.m(), "k": kind.k(), "grid": [kind.m(), u, 1],
        "u": u, "written_slots": exp.written_slots, "twin_rows": exp.twin_rows, "twin_selfcheck_mismatch": exp.selfcheck_bad,
        "launcher_plan_checked": plan_checked, "launcher_path": launcher_path,
        "mismatch_count": mm.count, "first_mismatches": mm.first,
        "words_compared": n_words * if kind.has_y1() { 2 } else { 1 },
        "x_row_stride": g.xrs, "y_slot_stride": g.ys, "x_src_rows": g.x_src_rows,
    });

    // Slab scratch path: the planner-format descriptor in a lane-union scratch (3 buckets when it fits, else 2).
    if ctx.slab_ready && case.tight && case.desc_in_contract() {
        row["slab_scratch_path"] = run_slab(ctx, case, kind, slabs, &exp, t)?;
    }
    Ok(row)
}

fn planner_words(desc: &Desc, s: usize, buckets: usize, bucket: usize) -> Vec<u32> {
    let mut w = vec![0u32; union_words(s, buckets)];
    let (ib, ob, sb) = union_bucket_words(s, buckets, bucket);
    let n = *desc.offs.last().unwrap();
    for (i, &id) in desc.ids.iter().enumerate() {
        w[ib + i] = id;
    }
    for i in 0..=s {
        w[ob + i] = if i < desc.offs.len() { desc.offs[i] } else { n };
    }
    for (i, &sl) in desc.slots.iter().take(n as usize).enumerate() {
        w[sb + i] = sl;
    }
    w
}

fn run_slab(
    ctx: &mut Ctx,
    case: &Case,
    kind: E8pKind,
    slabs: &BTreeMap<u32, (u64, u64)>,
    exp: &Expected,
    t: &mut Totals,
) -> Result<Value> {
    let c = case.c;
    let s = TOP_K * c;
    let g = case.geom[kind.index()];
    let cap = ctx
        .gpu
        .scratch
        .qwen4_slab_union_scratch
        .as_ref()
        .map_or(0, |b| b.size());
    let buckets = if union_words(s, 3) * 4 <= cap { 3 } else { 2 };
    let bucket = (case.name.len() + c + kind.index()) % buckets;
    let words = planner_words(&case.desc, s, buckets, bucket);
    {
        let ub = ctx
            .gpu
            .scratch
            .qwen4_slab_union_scratch
            .as_ref()
            .ok_or("slab union scratch missing")?;
        ctx.gpu
            .hip
            .memcpy_htod(ub, &u32_bytes(&words))
            .map_err(err)?;
    }
    let mut table = vec![0u64; N_EXPERTS];
    for (&id, &(gp, dp)) in slabs {
        table[id as usize] = if kind == E8pKind::GateUp { gp } else { dp };
    }
    let rows = kind.x_src_rows(c);
    let y_words = s * kind.y_slot_stride();
    let wp = ctx
        .gpu
        .alloc_tensor(&[N_EXPERTS * 8], DType::Raw)
        .map_err(err)?;
    let xt = ctx
        .gpu
        .alloc_tensor(&[rows * kind.k() * 4], DType::Raw)
        .map_err(err)?;
    let y0 = ctx
        .gpu
        .alloc_tensor(&[y_words * 4], DType::Raw)
        .map_err(err)?;
    let y1 = if kind.has_y1() {
        Some(
            ctx.gpu
                .alloc_tensor(&[y_words * 4], DType::Raw)
                .map_err(err)?,
        )
    } else {
        None
    };
    ctx.gpu
        .hip
        .memcpy_htod(&wp.buf, &u64_bytes(&table))
        .map_err(err)?;
    ctx.gpu
        .hip
        .memcpy_htod(
            &xt.buf,
            &f32_bytes(&case.x[kind.index()][..rows * kind.k()]),
        )
        .map_err(err)?;
    for yt in std::iter::once(&y0).chain(y1.as_ref()) {
        ctx.gpu
            .hip
            .memset(&yt.buf, 0xa5, yt.buf.size())
            .map_err(err)?;
    }
    ctx.gpu
        .e8p_i4_moe_gemv_slab(
            kind == E8pKind::Down,
            buckets,
            bucket,
            &wp,
            &xt,
            &y0,
            y1.as_ref(),
            c,
        )
        .map_err(err)?;
    ctx.gpu.hip.device_synchronize().map_err(err)?;
    let mut mm = Mm::default();
    let mut pad = 0u64;
    let mut g0 = vec![0u8; y_words * 4];
    ctx.gpu.hip.memcpy_dtoh(&mut g0, &y0.buf).map_err(err)?;
    compare(
        "y0",
        &g0,
        &exp.y0[..y_words],
        g.ys,
        kind.out_rows(),
        s,
        &mut mm,
        &mut pad,
    );
    if let (Some(yt), Some(e1)) = (y1.as_ref(), exp.y1.as_ref()) {
        let mut g1 = vec![0u8; y_words * 4];
        ctx.gpu.hip.memcpy_dtoh(&mut g1, &yt.buf).map_err(err)?;
        compare(
            "y1",
            &g1,
            &e1[..y_words],
            g.ys,
            kind.out_rows(),
            s,
            &mut mm,
            &mut pad,
        );
    }
    for tensor in [wp, xt, y0].into_iter().chain(y1) {
        ctx.gpu.free_tensor(tensor).map_err(err)?;
    }
    t.slab_runs += 1;
    t.slab_mismatches += mm.count;
    Ok(
        json!({"buckets": buckets, "bucket": bucket, "union_words": words.len(), "mismatch_count": mm.count, "first_mismatches": mm.first}),
    )
}

fn run_case(
    ctx: &mut Ctx,
    case: &Case,
    slabs_shared: Option<&BTreeMap<u32, (u64, u64)>>,
    shared_blobs: &[BlobAlloc],
    shared_dev: Option<&mut Dev>,
    t: &mut Totals,
) -> Result<Value> {
    t.cases += 1;
    let mut own = Dev::default();
    let mut own_blobs: Vec<BlobAlloc> = Vec::new();
    let mut own_slabs: BTreeMap<u32, (u64, u64)> = BTreeMap::new();
    if slabs_shared.is_none() {
        // Shuffled allocation order between random fillers: pointers are neither monotone nor uniformly strided.
        let mut rng = Rng::new(name_seed(&case.name, 0xa110c));
        let mut order: Vec<u32> = case.experts.keys().copied().collect();
        rng.shuffle(&mut order);
        for id in order {
            if rng.below(3) != 0 {
                own.alloc(&ctx.gpu.hip, 256 * (1 + rng.below(300)))?;
            }
            let e = &case.experts[&id];
            let gb = upload_blob(&mut own, &ctx.gpu.hip, &e.gate)?;
            let db = upload_blob(&mut own, &ctx.gpu.hip, &e.down)?;
            own_slabs.insert(id, (gb.buf.ptr, db.buf.ptr));
            own_blobs.push(gb);
            own_blobs.push(db);
        }
    }
    let (slabs, blobs): (&BTreeMap<u32, (u64, u64)>, &[BlobAlloc]) = match slabs_shared {
        Some(s) => (s, shared_blobs),
        None => (&own_slabs, &own_blobs),
    };
    let mut kinds = Vec::new();
    let mut failure: Option<String> = None;
    let mut guards = (0u64, 0u64);
    {
        let dev: &mut Dev = match shared_dev {
            Some(d) => d,
            None => &mut own,
        };
        for kind in E8pKind::ALL {
            match run_kind(ctx, case, kind, slabs, dev, t) {
                Ok(v) => kinds.push(v),
                Err(e) => {
                    failure = Some(e);
                    break;
                }
            }
        }
        if failure.is_none() {
            match check_guards(dev, &ctx.gpu.hip, blobs) {
                Ok(g) => guards = g,
                Err(e) => failure = Some(e),
            }
        }
    }
    own.release(&ctx.gpu.hip);
    if let Some(e) = failure {
        return Err(e);
    }
    t.guard_regions += guards.0;
    t.guard_changed += guards.1;
    let mm: u64 = kinds
        .iter()
        .map(|k| k["mismatch_count"].as_u64().unwrap_or(0))
        .sum();
    Ok(
        json!({"name": case.name, "origin": case.origin, "c": case.c, "slots": case.slots(), "unique_experts": case.experts.len(),
        "input_sha256": case.input_sha256(), "tight_geometry": case.tight, "mismatch_count": mm, "kinds": kinds,
        "guard_regions_checked": guards.0, "guard_regions_changed": guards.1}),
    )
}

// ---------------------------------------------------------------------------------------------------------------
// Real records (existing solved layer 0, pinned; repack of hipfire-arch-qwen4/examples/e8p_gp/records.rs)
// ---------------------------------------------------------------------------------------------------------------

struct Records {
    experts: Vec<Arc<ExpertBlobs>>,
    meta: Value,
}

fn load_records(cfg: &Cfg) -> Result<Records> {
    let dir = Path::new(
        cfg.records_dir
            .as_deref()
            .ok_or("--records-dir is required")?,
    );
    let manifest_path = Path::new(cfg.manifest.as_deref().ok_or("--manifest is required")?);
    let mraw =
        std::fs::read(manifest_path).map_err(|e| format!("{}: {e}", manifest_path.display()))?;
    let msha = sha_hex(&mraw);
    if msha != cfg.manifest_sha {
        return Err(format!(
            "{}: sha256 {msha} != pinned {}",
            manifest_path.display(),
            cfg.manifest_sha
        ));
    }
    let mj: Value = serde_json::from_slice(&mraw).map_err(err)?;
    if mj["arm"].as_str() != Some("vq-e8p") || mj["book"].as_str() != Some("e8p") {
        return Err(format!(
            "manifest arm/book {:?}/{:?} != vq-e8p/e8p",
            mj["arm"], mj["book"]
        ));
    }
    if mj["books_sha256"].as_str() != Some(e8p::E8P_BOOKS_SHA256) {
        return Err(format!(
            "manifest books_sha256 {:?} != K-E0 {}",
            mj["books_sha256"],
            e8p::E8P_BOOKS_SHA256
        ));
    }
    if mj["guard"]
        .as_array()
        .map(|a| !a.is_empty())
        .unwrap_or(true)
        || mj["refits"].as_u64() != Some(0)
    {
        return Err("manifest guard is not an empty list or refits != 0".into());
    }
    let mut per_proj: Vec<Vec<(Vec<u8>, String)>> = Vec::new();
    let mut files = serde_json::Map::new();
    for (kind, pin) in [
        (E8pKind::GateUp, cfg.gate_sha.as_str()),
        (E8pKind::Down, cfg.down_sha.as_str()),
    ] {
        let (m, k) = (kind.m(), kind.k());
        let (nb, ns) = (m * k / 8, m * k / 128);
        let want = 8 + 2 * nb + 2 * ns;
        let key = format!(
            "L{}.{}",
            cfg.layer,
            if kind == E8pKind::GateUp {
                "gate_up_proj"
            } else {
                "down_proj"
            }
        );
        let entry = &mj["files"][&key];
        if entry.is_null() {
            return Err(format!("manifest has no `{key}`"));
        }
        let shape: Vec<u64> = entry["expert_shape"]
            .as_array()
            .ok_or("manifest: expert_shape")?
            .iter()
            .filter_map(|x| x.as_u64())
            .collect();
        let sizes: Vec<u64> = entry["sizes"]
            .as_array()
            .ok_or("manifest: sizes")?
            .iter()
            .filter_map(|x| x.as_u64())
            .collect();
        let offsets: Vec<u64> = entry["offsets"]
            .as_array()
            .ok_or("manifest: offsets")?
            .iter()
            .filter_map(|x| x.as_u64())
            .collect();
        let src: Vec<&str> = entry["src"]
            .as_array()
            .ok_or("manifest: src")?
            .iter()
            .filter_map(|x| x.as_str())
            .collect();
        if shape != [m as u64, k as u64]
            || sizes.len() != N_EXPERTS
            || offsets.len() != N_EXPERTS
            || sizes.iter().any(|&s| s != want as u64)
            || offsets
                .iter()
                .enumerate()
                .any(|(i, &o)| o != (i * want) as u64)
            || src.len() != N_EXPERTS
            || src.iter().any(|s| *s != "vq")
            || entry["sha256"].as_str() != Some(pin)
            || entry["bytes"].as_u64() != Some((N_EXPERTS * want) as u64)
        {
            return Err(format!("{key}: manifest shape/sizes/offsets/src/sha256/bytes do not match the pinned record layout"));
        }
        let path = dir.join(format!("{key}.bin"));
        eprintln!("[real] reading {}", path.display());
        let raw = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        if raw.len() != N_EXPERTS * want {
            return Err(format!(
                "{}: {} bytes, expected {}",
                path.display(),
                raw.len(),
                N_EXPERTS * want
            ));
        }
        let file_sha = sha_hex(&raw);
        if file_sha != pin {
            return Err(format!(
                "{}: sha256 {file_sha} != pinned {pin}",
                path.display()
            ));
        }
        let outs: Vec<(Vec<u8>, String, [u64; 4])> = (0..N_EXPERTS)
            .into_par_iter()
            .map(|e| -> Result<(Vec<u8>, String, [u64; 4])> {
                let rec = &raw[e * want..(e + 1) * want];
                if rec[..8] != RECORD_HEADER {
                    return Err(format!("{key} expert {e}: bad record header"));
                }
                let (idx, signs, dbytes) = (
                    &rec[8..8 + nb],
                    &rec[8 + nb..8 + 2 * nb],
                    &rec[8 + 2 * nb..],
                );
                let mut blob = Vec::with_capacity(kind.blob_bytes());
                for i in 0..nb {
                    blob.extend_from_slice(
                        &(u16::from(idx[i]) | (u16::from(signs[i]) << 8)).to_le_bytes(),
                    );
                }
                blob.extend_from_slice(dbytes);
                if blob.len() != kind.blob_bytes() {
                    return Err(format!(
                        "{key} expert {e}: blob {} B != {}",
                        blob.len(),
                        kind.blob_bytes()
                    ));
                }
                let mut shift_words = 0u64;
                for i in 0..nb {
                    let w = u16::from_le_bytes([blob[2 * i], blob[2 * i + 1]]);
                    let (ix, s7, sh) = e8p::unpack_word_fields(w);
                    if ix != idx[i]
                        || (s7 | (u8::from(sh) << 7)) != signs[i]
                        || e8p::pack_word(ix, s7, sh) != w
                    {
                        return Err(format!(
                            "{key} expert {e} word {i}: repack inverse mismatch"
                        ));
                    }
                    shift_words += u64::from(sh);
                }
                let (mut rejects, mut zero) = (0u64, 0u64);
                let (mut dmin, mut dmax) = (u16::MAX, 0u16);
                for ch in dbytes.chunks_exact(2) {
                    let d = u16::from_le_bytes([ch[0], ch[1]]);
                    rejects += e8p::writer_guard(d).is_err() as u64;
                    zero += (d & 0x7fff == 0) as u64;
                    dmin = dmin.min(d);
                    dmax = dmax.max(d);
                }
                if rejects != 0 {
                    return Err(format!(
                        "{key} expert {e}: {rejects} scales fail the writer guard"
                    ));
                }
                Ok((
                    blob,
                    sha_hex(rec),
                    [shift_words, zero, dmin as u64, dmax as u64],
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let shift: u64 = outs.iter().map(|o| o.2[0]).sum();
        files.insert(
            kind_name(kind).into(),
            json!({"manifest_key": key, "path": path.display().to_string(), "bytes": raw.len(), "sha256": file_sha, "pinned_sha256": pin,
                "record_bytes": want, "blob_bytes": kind.blob_bytes(), "experts": N_EXPERTS, "words": N_EXPERTS * nb, "shift_words": shift,
                "scales": N_EXPERTS * ns, "writer_guard_rejects": 0, "zero_scale_groups": outs.iter().map(|o| o.2[1]).sum::<u64>(),
                "scale_bits_min": format!("{:#06x}", outs.iter().map(|o| o.2[2]).min().unwrap_or(0)),
                "scale_bits_max": format!("{:#06x}", outs.iter().map(|o| o.2[3]).max().unwrap_or(0)),
                "record_sha256_first": outs[0].1, "record_sha256_last": outs[N_EXPERTS - 1].1}),
        );
        per_proj.push(outs.into_iter().map(|o| (o.0, o.1)).collect());
    }
    let down = per_proj.pop().unwrap();
    let gate = per_proj.pop().unwrap();
    let experts: Vec<Arc<ExpertBlobs>> = gate
        .into_iter()
        .zip(down)
        .map(|(g, d)| Arc::new(ExpertBlobs::new(g.0, d.0)))
        .collect();
    let all: String = experts
        .iter()
        .map(|e| format!("{}{}", e.gate_sha, e.down_sha))
        .collect();
    let meta = json!({"layer": cfg.layer, "kind": "existing solved E8P records (vq-e8p arm), all 512 experts; repacked on CPU, not re-encoded",
        "manifest": {"path": manifest_path.display().to_string(), "sha256": msha, "pinned_sha256": cfg.manifest_sha, "books_sha256": mj["books_sha256"]},
        "files": Value::Object(files), "blob_sha256_all_experts_digest": sha_hex(all.as_bytes())});
    Ok(Records { experts, meta })
}

fn real_cases(rec: &Records, seed: u64) -> Result<Vec<Case>> {
    let experts_of = |ids: &[u32]| -> Experts {
        ids.iter()
            .map(|&id| (id, rec.experts[id as usize].clone()))
            .collect()
    };
    let mut out = Vec::new();
    // Cover cases: lane l of case k takes experts (k*80 + l*10 + j) mod 512; seven cases visit all 512.
    for k in 0..7usize {
        let route: Vec<u32> = (0..80).map(|i| ((k * 80 + i) % N_EXPERTS) as u32).collect();
        out.push(build_case(
            &format!("real_cover_{k}"),
            "real_records_layer",
            8,
            &route,
            None,
            experts_of(&route),
            CaseOpts::default(),
            seed,
        )?);
    }
    let base: Vec<u32> = (100..110).collect();
    let shared: Vec<u32> = (0..8).flat_map(|_| base.clone()).collect();
    out.push(build_case(
        "real_shared_c8",
        "real_records_layer",
        8,
        &shared,
        None,
        experts_of(&base),
        CaseOpts::default(),
        seed,
    )?);
    let edge: Vec<u32> = (0..5).chain(507..512).collect();
    let edge2: Vec<u32> = edge.iter().chain(edge.iter()).copied().collect();
    out.push(build_case(
        "real_edge_c2",
        "real_records_layer",
        2,
        &edge2,
        None,
        experts_of(&edge),
        CaseOpts::default(),
        seed,
    )?);
    let mut rng = Rng::new(name_seed("real_mixed_c8", seed));
    let pool = pick_ids(&mut rng, 20, true);
    let mut route = Vec::new();
    for _ in 0..8 {
        let mut p = pool.clone();
        rng.shuffle(&mut p);
        route.extend_from_slice(&p[..TOP_K]);
    }
    out.push(build_case(
        "real_mixed_c8",
        "real_records_layer",
        8,
        &route,
        None,
        experts_of(&pool),
        CaseOpts::default(),
        seed,
    )?);
    Ok(out)
}

// ---------------------------------------------------------------------------------------------------------------
// Timing: new E8P module vs the frozen MQ4 F4 module
// ---------------------------------------------------------------------------------------------------------------

fn mq4_blob(kind: E8pKind, id: u32) -> Vec<u8> {
    let mut rng = Rng::new(mix64(
        0x6d71_3400 ^ ((id as u64) << 8) ^ kind.index() as u64,
    ));
    let (m, k) = (kind.m(), kind.k());
    let (group_bytes, groups, hdr) = match kind {
        E8pKind::GateUp => (136usize, k / 256, 8usize),
        E8pKind::Down => (68, k / 128, 4),
    };
    let mut out = vec![0u8; m * groups * group_bytes];
    for gi in 0..m * groups {
        let base = gi * group_bytes;
        for h in 0..hdr / 4 {
            let sc = (((8 + rng.below(8)) as u16) << 10) | (rng.u32() as u16 & 0x3ff);
            let zp = half::f16::from_f32(-8.0 * half::f16::from_bits(sc).to_f32()).to_bits();
            out[base + 4 * h..base + 4 * h + 2].copy_from_slice(&sc.to_le_bytes());
            out[base + 4 * h + 2..base + 4 * h + 4].copy_from_slice(&zp.to_le_bytes());
        }
        let mut j = hdr;
        while j + 4 <= group_bytes {
            out[base + j..base + j + 4].copy_from_slice(&rng.u32().to_le_bytes());
            j += 4;
        }
    }
    out
}

fn stats(v: &[f64]) -> Value {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let at = |p: f64| s[((s.len() - 1) as f64 * p).round() as usize];
    json!({"n": s.len(), "min": s[0], "p10": at(0.1), "median": at(0.5), "p90": at(0.9), "max": s[s.len() - 1]})
}
fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}

struct LaunchSet {
    ptrs: [u64; 6],
    u: usize,
    tight: Geom,
}

fn run_timing(
    ctx: &mut Ctx,
    cfg: &Cfg,
    pool: &[Arc<ExpertBlobs>],
    t: &mut Totals,
    mq4_image: &[u8],
) -> Result<Value> {
    let np = pool.len();
    let hip_id = ctx.gpu.hip.current_device().map_err(err)?;
    let _ = hip_id;
    let mut dev = Dev::default();
    // Pools: E8P (guarded blobs) and synthetic MQ4, both resident for the whole run.
    eprintln!("[timing] uploading {np} E8P experts and {np} MQ4 experts");
    let mut blobs: Vec<BlobAlloc> = Vec::new();
    let mut slabs: BTreeMap<u32, (u64, u64)> = BTreeMap::new();
    let mut tables: [[Vec<u64>; 2]; 2] = [
        [vec![0; N_EXPERTS], vec![0; N_EXPERTS]],
        [vec![0; N_EXPERTS], vec![0; N_EXPERTS]],
    ];
    for (id, e) in pool.iter().enumerate() {
        let gb = upload_blob(&mut dev, &ctx.gpu.hip, &e.gate)?;
        let db = upload_blob(&mut dev, &ctx.gpu.hip, &e.down)?;
        slabs.insert(id as u32, (gb.buf.ptr, db.buf.ptr));
        tables[0][0][id] = gb.buf.ptr;
        tables[0][1][id] = db.buf.ptr;
        blobs.push(gb);
        blobs.push(db);
        let (mg, md) = (
            mq4_blob(E8pKind::GateUp, id as u32),
            mq4_blob(E8pKind::Down, id as u32),
        );
        tables[1][0][id] = dev.upload(&ctx.gpu.hip, &mg)?.ptr;
        tables[1][1][id] = dev.upload(&ctx.gpu.hip, &md)?.ptr;
    }
    let table_bufs: Vec<[Buf; 2]> = {
        let mut v = Vec::new();
        for codec in 0..2 {
            let g = dev.upload(&ctx.gpu.hip, &u64_bytes(&tables[codec][0]))?;
            let d = dev.upload(&ctx.gpu.hip, &u64_bytes(&tables[codec][1]))?;
            v.push([g, d]);
        }
        v
    };
    let mq4_module = ctx.gpu.hip.module_load_data(mq4_image).map_err(err)?;
    let e8p_module = ctx.gpu.hip.module_load_data(&ctx.image).map_err(err)?;
    let mut funcs: Vec<[Function; 2]> = Vec::new();
    for (module, syms) in [
        (&e8p_module, E8P_I4_MOE_GEMV_SYMBOLS),
        (&mq4_module, MQ4_SYMBOLS),
    ] {
        let mut pair = Vec::new();
        for s in syms {
            pair.push(
                ctx.gpu
                    .hip
                    .module_get_function(module, s)
                    .map_err(|e| format!("symbol {s}: {e}"))?,
            );
        }
        let b = pair.pop().unwrap();
        let a = pair.pop().unwrap();
        funcs.push([a, b]);
    }
    let evict_bytes = cfg.evict_mb << 20;
    let ev_a = dev.fill(&ctx.gpu.hip, evict_bytes, 0x5a)?;
    let ev_b = dev.fill(&ctx.gpu.hip, evict_bytes, 0x00)?;
    let start = ctx.gpu.hip.event_create().map_err(err)?;
    let stop = ctx.gpu.hip.event_create().map_err(err)?;

    let mut runs: Vec<Value> = Vec::new();
    let mut pairs: Vec<Value> = Vec::new();
    let mut verified: Vec<Value> = Vec::new();
    for &c in &cfg.cs {
        for scenario in ["distinct", "shared"] {
            if scenario == "shared" && c == 1 {
                continue;
            }
            let uniq = if scenario == "distinct" {
                TOP_K * c
            } else {
                TOP_K
            };
            if uniq > np {
                return Err(format!(
                    "--pool-experts {np} < {uniq} distinct experts of C{c} {scenario}"
                ));
            }
            let route_of = |set: usize| -> Vec<u32> {
                (0..c)
                    .flat_map(|l| {
                        (0..TOP_K).map(move |j| {
                            (if scenario == "distinct" {
                                set * uniq + l * TOP_K + j
                            } else {
                                set * TOP_K + j
                            }) % np
                        })
                    })
                    .map(|e| e as u32)
                    .collect()
            };
            let nsets_cold = (np / uniq).clamp(1, cfg.batch.max(1));
            for kind in E8pKind::ALL {
                // Byte-exact verification of the timed E8P route (set 0) against the K-E0 twin.
                let route0 = route_of(0);
                let sub: Experts = route0
                    .iter()
                    .map(|&id| (id, pool[id as usize].clone()))
                    .collect();
                let case = build_case(
                    &format!("timing_c{c}_{scenario}"),
                    "timing_verify",
                    c,
                    &route0,
                    None,
                    sub,
                    CaseOpts::default(),
                    cfg.seed,
                )?;
                let mut vd = Dev::default();
                let before = t.mismatches;
                let row = run_kind(ctx, &case, kind, &slabs, &mut vd, t);
                vd.release(&ctx.gpu.hip);
                let row = row?;
                if t.mismatches != before {
                    return Err(format!("timing_c{c}_{scenario} {}: E8P not byte exact; refusing to time a wrong kernel", kind_name(kind)));
                }
                verified.push(json!({"c": c, "scenario": scenario, "kind": kind_name(kind), "mismatch_count": row["mismatch_count"], "words_compared": row["words_compared"]}));

                // Launch sets (descriptors, X, Y per set), shared by both codecs except the pointer table.
                let g = Geom {
                    xrs: kind.k(),
                    ys: kind.y_slot_stride(),
                    x_src_rows: kind.x_src_rows(c),
                };
                let mut sets: Vec<LaunchSet> = Vec::new();
                for set in 0..nsets_cold {
                    let route = route_of(set);
                    let u = E8pUnionHost::from_route(&route, |_| true).map_err(err)?;
                    let mut rng = Rng::new(name_seed("timing_x", set as u64 + 17 * c as u64));
                    let rows = kind.x_src_rows(c);
                    let mut x = Vec::with_capacity(rows * kind.k());
                    for r in 0..rows {
                        x.extend(gen_x_row(r % 2, kind.k(), &mut rng));
                    }
                    let (ib, ob, sb) = (
                        dev.upload(&ctx.gpu.hip, &u32_bytes(&u.expert_ids))?,
                        dev.upload(&ctx.gpu.hip, &u32_bytes(&u.expert_offsets))?,
                        dev.upload(&ctx.gpu.hip, &u32_bytes(&u.slot_ids))?,
                    );
                    let xb = dev.upload(&ctx.gpu.hip, &f32_bytes(&x))?;
                    let yw = TOP_K * c * kind.y_slot_stride() * 4;
                    let y0 = dev.fill(&ctx.gpu.hip, yw, 0xa5)?;
                    let y1 = if kind.has_y1() {
                        dev.fill(&ctx.gpu.hip, yw, 0xa5)?
                    } else {
                        Buf {
                            idx: 0,
                            ptr: 0,
                            len: 0,
                        }
                    };
                    sets.push(LaunchSet {
                        ptrs: [ib.ptr, ob.ptr, sb.ptr, xb.ptr, y0.ptr, y1.ptr],
                        u: u.expert_ids.len(),
                        tight: g,
                    });
                }
                let args_of = |codec: usize, s: &LaunchSet| -> [u8; KERNARG_BYTES] {
                    let tb = table_bufs[codec][kind.index()];
                    kernargs_of(
                        kind,
                        c,
                        s.u,
                        [
                            tb.ptr, s.ptrs[0], s.ptrs[1], s.ptrs[2], s.ptrs[3], s.ptrs[4],
                            s.ptrs[5],
                        ],
                        s.tight,
                    )
                    .to_bytes()
                };
                let slots = TOP_K * c;
                for mode in ["resident", "cold"] {
                    if (mode == "resident" && !cfg.resident) || (mode == "cold" && !cfg.cold) {
                        continue;
                    }
                    let b = if mode == "cold" {
                        nsets_cold
                    } else {
                        cfg.batch
                    };
                    let mut samples: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
                    for rep in 0..cfg.warmup + cfg.reps {
                        let order = if rep % 2 == 0 { [0usize, 1] } else { [1, 0] };
                        for codec in order {
                            if mode == "cold" {
                                ctx.gpu
                                    .hip
                                    .memcpy_dtod(
                                        &dev.bufs[ev_b.idx],
                                        &dev.bufs[ev_a.idx],
                                        evict_bytes,
                                    )
                                    .map_err(err)?;
                            }
                            ctx.gpu.hip.device_synchronize().map_err(err)?;
                            ctx.gpu.hip.event_record(&start, None).map_err(err)?;
                            for li in 0..b {
                                let s = &sets[if mode == "cold" { li } else { 0 }];
                                let mut a = args_of(codec, s);
                                // SAFETY: exact 96-byte F4 kernarg of the symbol; every pointer is a live allocation of `dev`
                                // until `release` (after the final synchronize).
                                unsafe {
                                    ctx.gpu.hip.launch_kernel_blob(
                                        &funcs[codec][kind.index()],
                                        [kind.m() as u32, s.u as u32, 1],
                                        [32, 1, 1],
                                        0,
                                        None,
                                        &mut a,
                                    )
                                }
                                .map_err(err)?;
                            }
                            ctx.gpu.hip.event_record(&stop, None).map_err(err)?;
                            ctx.gpu.hip.event_synchronize(&stop).map_err(err)?;
                            let ms = ctx.gpu.hip.event_elapsed_ms(&start, &stop).map_err(err)?;
                            if rep >= cfg.warmup {
                                samples[codec].push(ms as f64 * 1000.0 / b as f64);
                            }
                            t.launches += b as u64;
                        }
                    }
                    let rows_per_launch = kind.m() * slots;
                    let med = [median(&samples[0]), median(&samples[1])];
                    for (codec, name) in ["e8p", "mq4"].iter().enumerate() {
                        runs.push(json!({"codec": name, "symbol": if codec == 0 { kind.symbol() } else { MQ4_SYMBOLS[kind.index()] },
                            "kind": kind_name(kind), "c": c, "scenario": scenario, "mode": mode, "slots": slots, "unique_experts": uniq,
                            "m": kind.m(), "k": kind.k(), "rows_per_launch": rows_per_launch, "launches_per_sample": b,
                            "launch_us": stats(&samples[codec]), "median_row_us": med[codec] / rows_per_launch as f64,
                            "samples_launch_us": samples[codec]}));
                    }
                    pairs.push(json!({"kind": kind_name(kind), "c": c, "scenario": scenario, "mode": mode,
                        "e8p_median_launch_us": med[0], "mq4_median_launch_us": med[1],
                        "e8p_row_us": med[0] / rows_per_launch as f64, "mq4_row_us": med[1] / rows_per_launch as f64,
                        "ratio_e8p_over_mq4": med[0] / med[1], "mq4_over_e8p_speed": med[1] / med[0]}));
                    eprintln!("[timing] C{c} {scenario} {} {mode}: e8p {:.2} us, mq4 {:.2} us, ratio {:.4}", kind_name(kind), med[0], med[1], med[0] / med[1]);
                }
            }
        }
    }
    ctx.gpu.hip.event_destroy(start).map_err(err)?;
    ctx.gpu.hip.event_destroy(stop).map_err(err)?;
    let (checked, changed) = check_guards(&dev, &ctx.gpu.hip, &blobs)?;
    t.guard_regions += checked;
    t.guard_changed += changed;
    dev.release(&ctx.gpu.hip);
    Ok(
        json!({"pairs": pairs, "runs": runs, "verified_before_timing": verified, "pool_experts": np,
        "evict_bytes": evict_bytes, "eviction": "device-to-device copy of the evict buffer before the start event (reads and writes evict_bytes; same for both codecs)",
        "normalization": "launch_us = hipEvent elapsed over `launches_per_sample` back-to-back launches / launches_per_sample (launch-inclusive, no empty-kernel floor); row_us = launch_us / (M * slots), slots = 10 C",
        "scope": "e8p vs frozen mq4sym F4 symbols of mq_i4_moe_gemv_pm_gfx1201, same kernarg/descriptors/X/Y; mq4 data synthetic (random legacy-wire codes/scales); claim bar only, no kill threshold"}),
    )
}

// ---------------------------------------------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    Synthetic,
    Real,
    Timing,
}

struct Cfg {
    mode: Mode,
    e8p_co: String,
    mq4_co: Option<String>,
    e8p_co_sha: Option<String>,
    mq4_co_sha: Option<String>,
    out: Option<String>,
    overwrite: bool,
    seed: u64,
    only: Vec<String>,
    device: i32,
    card: Option<String>,
    expect_uuid: Option<String>,
    expect_bdf: Option<String>,
    expect_arch: String,
    target: E8pArch,
    records_dir: Option<String>,
    manifest: Option<String>,
    layer: usize,
    manifest_sha: String,
    gate_sha: String,
    down_sha: String,
    reps: usize,
    warmup: usize,
    batch: usize,
    evict_mb: usize,
    pool: usize,
    resident: bool,
    cold: bool,
    cs: Vec<usize>,
    allow_few_reps: bool,
    negative: bool,
    fabric: BTreeMap<&'static str, Option<String>>,
}

const USAGE: &str = "usage: e8p_i4_moe_gemv_oracle <synthetic|real|timing> --e8p-codeobject OBJ --out RECEIPT.json \
[--mq4-codeobject MQ4.hxaco (timing)] [--records-dir DIR --manifest FILE [--layer 0] (real; optional for timing)] \
[--seed N] [--only SUBSTR]... [--reps 25 --warmup 5 --batch 64 --evict-mb 256 --pool-experts 512 --modes resident,cold --cs 1,..,8 --allow-few-reps] \
[--device N] [--card LABEL] [--expect-uuid U] [--expect-bdf B (required for gfx1151)] [--expect-arch gfx1201|gfx1151] [--e8p-codeobject-sha256 H] [--mq4-codeobject-sha256 H] \
[--manifest-sha256 H --gate-sha256 H --down-sha256 H (layers other than 0)] [--negative-control] [--overwrite] \
[--fabric-class C --fabric-owner O --fabric-job J --fabric-card CARD --fabric-cmd STR]";

fn env_nonempty(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

fn parse_args(argv: &[String]) -> Result<Cfg> {
    let mode = match argv.first().map(String::as_str) {
        Some("synthetic") => Mode::Synthetic,
        Some("real") => Mode::Real,
        Some("timing") => Mode::Timing,
        _ => return Err(USAGE.into()),
    };
    let mut cfg = Cfg {
        mode,
        e8p_co: String::new(),
        mq4_co: None,
        e8p_co_sha: None,
        mq4_co_sha: None,
        out: None,
        overwrite: false,
        seed: 0,
        only: Vec::new(),
        device: 0,
        card: None,
        expect_uuid: None,
        expect_bdf: None,
        expect_arch: "gfx1201".into(),
        target: E8pArch::Gfx1201,
        records_dir: None,
        manifest: None,
        layer: 0,
        manifest_sha: MANIFEST_SHA256.into(),
        gate_sha: L0_GATE_SHA256.into(),
        down_sha: L0_DOWN_SHA256.into(),
        reps: 25,
        warmup: 5,
        batch: 64,
        evict_mb: 256,
        pool: N_EXPERTS,
        resident: true,
        cold: true,
        cs: (1..=MAX_LANES).collect(),
        allow_few_reps: false,
        negative: false,
        fabric: BTreeMap::new(),
    };
    let mut fab: BTreeMap<&'static str, Option<String>> = BTreeMap::new();
    for (key, var) in [
        ("class", "FABRIC_CLASS"),
        ("owner", "FABRIC_OWNER"),
        ("job", "FABRIC_JOB"),
        ("card", "FABRIC_CARD"),
        ("cmd", "FABRIC_CMD"),
    ] {
        fab.insert(key, env_nonempty(var));
    }
    let mut i = 1;
    let need = |i: &mut usize| -> Result<String> {
        *i += 1;
        argv.get(*i)
            .cloned()
            .ok_or_else(|| format!("{} needs a value\n{USAGE}", argv[*i - 1]))
    };
    let num = |s: String, flag: &str| -> Result<usize> {
        s.parse().map_err(|_| format!("{flag}: not a number"))
    };
    while i < argv.len() {
        match argv[i].as_str() {
            "--e8p-codeobject" => cfg.e8p_co = need(&mut i)?,
            "--mq4-codeobject" => cfg.mq4_co = Some(need(&mut i)?),
            "--e8p-codeobject-sha256" => cfg.e8p_co_sha = Some(need(&mut i)?),
            "--mq4-codeobject-sha256" => cfg.mq4_co_sha = Some(need(&mut i)?),
            "--out" => cfg.out = Some(need(&mut i)?),
            "--overwrite" => cfg.overwrite = true,
            "--seed" => cfg.seed = num(need(&mut i)?, "--seed")? as u64,
            "--only" => cfg.only.push(need(&mut i)?),
            "--device" => cfg.device = num(need(&mut i)?, "--device")? as i32,
            "--card" => cfg.card = Some(need(&mut i)?),
            "--expect-uuid" => cfg.expect_uuid = Some(need(&mut i)?),
            "--expect-bdf" => cfg.expect_bdf = Some(need(&mut i)?.to_ascii_lowercase()),
            "--expect-arch" => cfg.expect_arch = need(&mut i)?,
            "--records-dir" => cfg.records_dir = Some(need(&mut i)?),
            "--manifest" => cfg.manifest = Some(need(&mut i)?),
            "--layer" => cfg.layer = num(need(&mut i)?, "--layer")?,
            "--manifest-sha256" => cfg.manifest_sha = need(&mut i)?,
            "--gate-sha256" => cfg.gate_sha = need(&mut i)?,
            "--down-sha256" => cfg.down_sha = need(&mut i)?,
            "--reps" => cfg.reps = num(need(&mut i)?, "--reps")?,
            "--warmup" => cfg.warmup = num(need(&mut i)?, "--warmup")?,
            "--batch" => cfg.batch = num(need(&mut i)?, "--batch")?,
            "--evict-mb" => cfg.evict_mb = num(need(&mut i)?, "--evict-mb")?,
            "--pool-experts" => cfg.pool = num(need(&mut i)?, "--pool-experts")?,
            "--modes" => {
                let v = need(&mut i)?;
                cfg.resident = v.contains("resident");
                cfg.cold = v.contains("cold");
            }
            "--cs" => {
                cfg.cs = need(&mut i)?
                    .split(',')
                    .map(|s| {
                        s.trim()
                            .parse::<usize>()
                            .map_err(|_| "--cs: not a number list".to_string())
                    })
                    .collect::<Result<_>>()?;
                if cfg.cs.iter().any(|c| !(1..=MAX_LANES).contains(c)) {
                    return Err("--cs entries must be in 1..=8".into());
                }
            }
            "--allow-few-reps" => cfg.allow_few_reps = true,
            "--negative-control" => cfg.negative = true,
            "--fabric-class" => {
                fab.insert("class", Some(need(&mut i)?));
            }
            "--fabric-owner" => {
                fab.insert("owner", Some(need(&mut i)?));
            }
            "--fabric-job" => {
                fab.insert("job", Some(need(&mut i)?));
            }
            "--fabric-card" => {
                fab.insert("card", Some(need(&mut i)?));
            }
            "--fabric-cmd" => {
                fab.insert("cmd", Some(need(&mut i)?));
            }
            other => return Err(format!("unknown argument {other}\n{USAGE}")),
        }
        i += 1;
    }
    cfg.fabric = fab;
    cfg.target = E8pArch::of_name(&cfg.expect_arch).ok_or_else(|| {
        format!(
            "--expect-arch {:?}: this oracle serves gfx1201 and gfx1151\n{USAGE}",
            cfg.expect_arch
        )
    })?;
    if cfg.target == E8pArch::Gfx1151 && cfg.expect_bdf.is_none() {
        return Err(format!("gfx1151 has no ROCr UUID: --expect-bdf is required (--expect-uuid is optional)\n{USAGE}"));
    }
    if cfg.target == E8pArch::Gfx1151 && cfg.mode == Mode::Timing {
        return Err("timing is gfx1201 only: its baseline is the frozen mq_i4_moe_gemv_pm_gfx1201 MQ4 module (none exists for gfx1151)".into());
    }
    if cfg.e8p_co.is_empty() {
        return Err(format!("--e8p-codeobject is required\n{USAGE}"));
    }
    if cfg.mode == Mode::Timing {
        if cfg.mq4_co.is_none() {
            return Err(
                "timing needs --mq4-codeobject (the frozen mq_i4_moe_gemv_pm_gfx1201 bundle)"
                    .into(),
            );
        }
        if (cfg.reps < 20 || cfg.warmup < 3) && !cfg.allow_few_reps {
            return Err("timing needs --reps >= 20 and --warmup >= 3 (or --allow-few-reps)".into());
        }
        if cfg.batch == 0 || !(cfg.resident || cfg.cold) || cfg.pool < TOP_K || cfg.pool > N_EXPERTS
        {
            return Err("timing: bad --batch / --modes / --pool-experts".into());
        }
        if cfg.records_dir.is_some() != cfg.manifest.is_some() {
            return Err("--records-dir and --manifest go together".into());
        }
    }
    if cfg.mode == Mode::Real && (cfg.records_dir.is_none() || cfg.manifest.is_none()) {
        return Err("real mode needs --records-dir and --manifest".into());
    }
    if cfg.layer != 0 && (cfg.manifest_sha == MANIFEST_SHA256 && cfg.gate_sha == L0_GATE_SHA256) {
        return Err("--layer other than 0 needs explicit --gate-sha256/--down-sha256".into());
    }
    Ok(cfg)
}

fn selected(cfg: &Cfg, name: &str) -> bool {
    cfg.only.is_empty() || cfg.only.iter().any(|s| name.contains(s.as_str()))
}

fn load_image(path: &str, pin: &Option<String>) -> Result<(Arc<[u8]>, String)> {
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let sha = sha_hex(&bytes);
    if let Some(p) = pin {
        if !sha.eq_ignore_ascii_case(p) {
            return Err(format!("{path}: sha256 {sha} != pinned {p}"));
        }
    }
    Ok((Arc::from(bytes), sha))
}

/// K-E0 self-check before any launch: `unpack_word_swar == unpack_word_ref` over all 65536 wire words.
fn wire_selfcheck() -> Value {
    let (lut, mask) = (e8p::build_lut(), e8p::build_mask());
    let mut bad = 0u64;
    for w in 0..=65535u16 {
        let (idx, sign7, shift) = e8p::unpack_word_fields(w);
        bad += (e8p::unpack_word_swar(&lut, &mask, idx, sign7, shift)
            != e8p::unpack_word_ref(idx, sign7, shift)) as u64;
        bad += (e8p::pack_word(idx, sign7, shift) != w) as u64;
    }
    json!({"words": 65536, "mismatches": bad})
}

/// Distinct wire words and per-idx / sign / shift coverage of the `every_word` generator over the gate plane.
fn everyword_coverage() -> Value {
    let blob = gen_blob(E8pKind::GateUp, 1, WordMode::EveryWord, ScaleMode::Normal);
    let (words, _) = parse_blob(E8pKind::GateUp, &blob);
    let set: BTreeSet<u16> = words.iter().copied().collect();
    json!({"plane_words": words.len(), "distinct_words": set.len()})
}

fn run() -> Result<(Value, i32)> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let cfg = parse_args(&argv)?;
    if sha_hex(REFERENCE_SRC) != REFERENCE_SHA256 {
        return Err(format!(
            "e8p_i4_reference.rs sha256 {} != frozen {REFERENCE_SHA256}",
            sha_hex(REFERENCE_SRC)
        ));
    }
    if let Some(out) = &cfg.out {
        if Path::new(out).exists() && !cfg.overwrite {
            return Err(format!("{out} exists (pass --overwrite)"));
        }
    }
    // Inputs are pinned and hashed on the CPU before any device is touched.
    let (image, e8p_sha) = load_image(&cfg.e8p_co, &cfg.e8p_co_sha)?;
    let mq4 = match (&cfg.mq4_co, cfg.mode) {
        (Some(p), Mode::Timing) => Some(load_image(p, &cfg.mq4_co_sha)?),
        _ => None,
    };
    let records = match cfg.mode {
        Mode::Real => Some(load_records(&cfg)?),
        Mode::Timing if cfg.records_dir.is_some() => Some(load_records(&cfg)?),
        _ => None,
    };
    let selfcheck = wire_selfcheck();
    let coverage = everyword_coverage();

    let mut gpu = Gpu::init_with_device(cfg.device).map_err(err)?;
    let id = gpu.hip.current_device().map_err(err)?;
    let uuid = gpu.hip.device_uuid(id).map_err(err)?;
    let bdf = gpu
        .hip
        .device_pci_bus_id(id)
        .map_err(err)?
        .to_ascii_lowercase();
    let arch = gpu.hip.get_arch(id).map_err(err)?;
    let identity_match = arch == cfg.expect_arch
        && cfg
            .expect_uuid
            .as_ref()
            .map_or(true, |u| uuid.contains(u.trim_start_matches("GPU-")))
        && cfg.expect_bdf.as_ref().map_or(true, |b| &bdf == b);
    let device = json!({"index": id, "uuid": uuid, "bdf": bdf, "arch": arch, "card": cfg.card,
        "expected_uuid": cfg.expect_uuid, "expected_bdf": cfg.expect_bdf, "expected_arch": cfg.expect_arch,
        "uuid_authoritative": cfg.target != E8pArch::Gfx1151, "uuid_checked": cfg.expect_uuid.is_some(), "identity_match": identity_match});
    let fabric: serde_json::Map<String, Value> = cfg
        .fabric
        .iter()
        .map(|(k, v)| (k.to_string(), json!(v)))
        .collect();
    let mut receipt = json!({"schema": SCHEMA, "mode": format!("{:?}", cfg.mode).to_lowercase(),
        "command": {"argv": argv, "build_command": BUILD_COMMAND}, "fabric": Value::Object(fabric), "device": device,
        "harness_sha256": sha_hex(HARNESS_SRC), "reference_sha256": sha_hex(REFERENCE_SRC), "reference_sha256_pinned": REFERENCE_SHA256,
        "launcher_sha256": sha_hex(LAUNCHER_SRC),
        "codeobject": {"e8p": {"path": cfg.e8p_co, "sha256": e8p_sha, "bytes": image.len(), "symbols": cfg.target.symbols(), "module": cfg.target.module()},
            "mq4": mq4.as_ref().map(|(b, s)| json!({"path": cfg.mq4_co, "sha256": s, "bytes": b.len(), "symbols": MQ4_SYMBOLS}))},
        "wire_selfcheck_unpack_swar_vs_ref": selfcheck, "every_word_generator_coverage": coverage});
    if !identity_match {
        receipt["error"] = json!("device identity mismatch");
        receipt["pass"] = json!(false);
        receipt["rc"] = json!(1);
        return Ok((receipt, 1));
    }
    if selfcheck["mismatches"] != 0 {
        return Err(
            "K-E0 SWAR unpack disagrees with the reference over the 65536 wire words".into(),
        );
    }
    if coverage["distinct_words"] != 65536 {
        return Err("every_word generator does not cover all 65536 wire words".into());
    }

    let module = E8pMoeDecodeModule::load(&mut gpu, &image).map_err(err)?;
    let slab_error: Option<String> = gpu
        .ensure_e8p_i4_moe_gemv_kernels(image.clone())
        .and_then(|()| gpu.reserve_qwen4_slab(0))
        .err()
        .map(|e| e.to_string());
    let slab_ready = slab_error.is_none();
    let mut ctx = Ctx {
        gpu,
        module,
        image: image.clone(),
        negative: cfg.negative,
        slab_ready,
    };
    let mut t = Totals::default();
    let mut rows: Vec<Value> = Vec::new();
    let mut extra = serde_json::Map::new();
    extra.insert("slab_scratch_path_available".into(), json!(slab_ready));
    extra.insert("slab_scratch_path_error".into(), json!(slab_error));

    match cfg.mode {
        Mode::Synthetic => {
            let mut cache = ExpertCache::new();
            for name in SYNTH_CASES.iter().filter(|n| selected(&cfg, n)) {
                eprintln!("[synthetic] {name}");
                let case = build_synthetic(name, &mut cache, cfg.seed)?;
                rows.push(run_case(&mut ctx, &case, None, &[], None, &mut t)?);
                // Keep the cache bounded: experts of earlier cases are regenerated when reused.
                if cache.len() > 160 {
                    cache.clear();
                }
            }
        }
        Mode::Real => {
            let rec = records.as_ref().expect("loaded above");
            extra.insert("real_records_layer".into(), rec.meta.clone());
            // All 512 experts resident once, each blob exactly its wire size with 0xFF guards.
            let mut dev = Dev::default();
            let mut blobs: Vec<BlobAlloc> = Vec::new();
            let mut slabs: BTreeMap<u32, (u64, u64)> = BTreeMap::new();
            for (id, e) in rec.experts.iter().enumerate() {
                let gb = upload_blob(&mut dev, &ctx.gpu.hip, &e.gate)?;
                let db = upload_blob(&mut dev, &ctx.gpu.hip, &e.down)?;
                slabs.insert(id as u32, (gb.buf.ptr, db.buf.ptr));
                blobs.push(gb);
                blobs.push(db);
            }
            let cases = real_cases(rec, cfg.seed)?;
            let run = (|| -> Result<()> {
                for case in cases.iter().filter(|c| selected(&cfg, &c.name)) {
                    eprintln!("[real] {}", case.name);
                    rows.push(run_case(
                        &mut ctx,
                        case,
                        Some(&slabs),
                        &blobs,
                        Some(&mut dev),
                        &mut t,
                    )?);
                }
                Ok(())
            })();
            let (checked, changed) = check_guards(&dev, &ctx.gpu.hip, &blobs)?;
            t.guard_regions += checked;
            t.guard_changed += changed;
            dev.release(&ctx.gpu.hip);
            run?;
        }
        Mode::Timing => {
            let pool: Vec<Arc<ExpertBlobs>> = match &records {
                Some(r) => {
                    extra.insert("real_records_layer".into(), r.meta.clone());
                    r.experts.iter().take(cfg.pool).cloned().collect()
                }
                None => {
                    let ids: Vec<u32> = (0..cfg.pool as u32).collect();
                    let mut cache = ExpertCache::new();
                    let m = experts_for(&mut cache, &ids, &|_| WordMode::Random, &|_| {
                        ScaleMode::Normal
                    });
                    m.into_values().collect()
                }
            };
            let (mq4_image, _) = mq4.as_ref().expect("checked in parse_args");
            let timing = run_timing(&mut ctx, &cfg, &pool, &mut t, mq4_image)?;
            extra.insert("timing".into(), timing);
            extra.insert(
                "timing_e8p_data".into(),
                json!(if records.is_some() {
                    "real L0 records"
                } else {
                    "synthetic random wire"
                }),
            );
        }
    }

    let coverage_ok = cfg.mode != Mode::Real
        || !cfg.only.is_empty()
        || (t.experts_gate.len() == N_EXPERTS && t.experts_down.len() == N_EXPERTS);
    let exact = t.mismatches == 0
        && t.selfcheck_bad == 0
        && t.plan_mismatches == 0
        && t.slab_mismatches == 0
        && slab_ready
        && t.guard_changed == 0
        && coverage_ok
        && (t.kind_runs > 0);
    let rc = if exact { 0 } else { 3 };
    receipt["counts"] = json!({"cases": t.cases, "kind_runs": t.kind_runs, "launches": t.launches, "slab_scratch_runs": t.slab_runs,
        "twin_rows": t.twin_rows, "words_compared": t.words_compared, "pad_and_tail_words_compared": t.pad_words_compared,
        "launcher_plan_checks": t.plan_checks, "experts_verified_gate_up": t.experts_gate.len(), "experts_verified_down": t.experts_down.len()});
    receipt["mismatch_count"] = json!(t.mismatches);
    receipt["twin_selfcheck_mismatch_count"] = json!(t.selfcheck_bad);
    receipt["launcher_plan_mismatch_count"] = json!(t.plan_mismatches);
    receipt["slab_scratch_mismatch_count"] = json!(t.slab_mismatches);
    receipt["poison_checks"] = json!({"weight_guard_regions_checked": t.guard_regions, "weight_guard_regions_changed": t.guard_changed,
        "y_pad_and_tail_words_compared": t.pad_words_compared, "sentinel": bits_hex(SENTINEL_BITS), "x_pad_nan": bits_hex(NAN_BITS),
        "note": "a read of a guard/pad word is visible only through the output; address-census proof of exact u16 word/scale accesses is the kernel emitter's"});
    receipt["negative_control"] = json!(cfg.negative);
    receipt["cases"] = Value::Array(rows);
    for (k, v) in extra {
        receipt[k] = v;
    }
    receipt["real_coverage_complete"] = json!(coverage_ok);
    receipt["byte_exact"] = json!(exact);
    receipt["pass"] = json!(exact);
    receipt["rc"] = json!(rc);
    Ok((receipt, rc))
}

fn main() {
    match run() {
        Ok((v, rc)) => {
            let text = serde_json::to_string_pretty(&v).unwrap();
            if let Some(path) = std::env::args()
                .skip(1)
                .collect::<Vec<_>>()
                .windows(2)
                .find(|w| w[0] == "--out")
                .map(|w| w[1].clone())
            {
                if let Err(e) = std::fs::write(&path, &text) {
                    eprintln!("{path}: {e}");
                    std::process::exit(1);
                }
            }
            println!("{text}");
            if rc != 0 {
                eprintln!("E8P MoE GEMV oracle: NOT exact (rc {rc})");
                std::process::exit(rc);
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

// ---------------------------------------------------------------------------------------------------------------
// Local known-answer tests (no GPU):
//   cargo test -p hipfire-runtime --release --features lab --example e8p_i4_moe_gemv_oracle
// ---------------------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_word_generator_covers_all_65536_wire_words() {
        assert_eq!(everyword_coverage()["distinct_words"], 65536);
        assert_eq!(wire_selfcheck()["mismatches"], 0);
    }

    #[test]
    fn synthetic_blobs_have_the_wire_size_and_pass_the_writer_guard() {
        for kind in E8pKind::ALL {
            for (wm, sm) in [
                (WordMode::Extreme, ScaleMode::Edge),
                (WordMode::Parity, ScaleMode::Normal),
            ] {
                let blob = gen_blob(kind, 7, wm, sm);
                assert_eq!(blob.len(), kind.blob_bytes());
                let (_, scales) = parse_blob(kind, &blob);
                assert!(scales.iter().all(|&d| e8p::writer_guard(d).is_ok()));
                if sm == ScaleMode::Edge {
                    for edge in [0x0400u16, 0x0401, 0x0403, 0x0405, 0x7bff] {
                        assert!(scales.contains(&edge), "{edge:#06x} missing");
                    }
                }
            }
        }
    }

    #[test]
    fn extreme_generator_reaches_the_largest_decoded_magnitude() {
        let blob = gen_blob(E8pKind::Down, 2, WordMode::Extreme, ScaleMode::Normal);
        // Decode the real expert's entire word region, without copying words or
        // allocating a decoded matrix. A prefix need not cover the sign/shift cycle.
        let nw = E8pKind::Down.m() * E8pKind::Down.k() / 8;
        let max_magnitude = blob[..nw * 2]
            .chunks_exact(2)
            .flat_map(|bytes| {
                let word = u16::from_le_bytes([bytes[0], bytes[1]]);
                let (idx, sign7, shift) = e8p::unpack_word_fields(word);
                (0..2).flat_map(move |half| e8p::decode8_lanes(idx, sign7, shift, half))
            })
            .map(i8::abs)
            .max()
            .unwrap();
        assert!(max_magnitude >= 9);
        let max_abs2 = *e8p::E8P_ABS2.iter().flatten().max().unwrap() as i8;
        assert_eq!(max_magnitude, 2 * max_abs2 + 1);
    }

    #[test]
    fn descriptor_cases_follow_the_kernel_validation() {
        let mut cache = ExpertCache::new();
        let case = build_synthetic("c1_refused_n9_and_end", &mut cache, 0).unwrap();
        assert!(!case.desc_in_contract());
        let ids = case.desc.ids.clone();
        let (exp0, exp1) = (&ids, &ids);
        assert_eq!(exp0.len(), 3);
        assert_eq!(exp1.len(), 3);
        for name in ["c2_nohit_tail", "c1_empty_bucket"] {
            let c = build_synthetic(name, &mut cache, 0).unwrap();
            assert!(c.desc_in_contract() || name == "c1_empty_bucket", "{name}");
        }
        let tail = build_synthetic("c2_nohit_tail", &mut cache, 0).unwrap();
        assert_eq!(tail.desc.ids.len(), 20);
        assert_eq!(tail.desc.offs.len(), 21);
        assert_eq!(*tail.desc.offs.last().unwrap(), 20);
        let oob = build_synthetic("c4_x_oob", &mut cache, 0).unwrap();
        assert!(!oob.tight);
        assert_eq!(oob.geom[0].x_src_rows, 3);
        assert_eq!(oob.geom[1].x_src_rows, 30);
    }

    #[test]
    fn planner_words_use_the_incumbent_layout() {
        let desc = Desc {
            ids: vec![5, 9],
            offs: vec![0, 2, 3],
            slots: vec![0, 4, 7],
        };
        let w = planner_words(&desc, 10, 2, 1);
        let (ib, ob, sb) = union_bucket_words(10, 2, 1);
        assert_eq!((ib, ob, sb), (10, 31, 52));
        assert_eq!(&w[ib..ib + 3], &[5, 9, 0]);
        assert_eq!(&w[ob..ob + 4], &[0, 2, 3, 3]);
        assert_eq!(w[ob + 10], 3);
        assert_eq!(&w[sb..sb + 4], &[0, 4, 7, 0]);
        assert!(w[..10].iter().all(|&x| x == 0));
    }

    #[test]
    fn kernarg_layout_is_the_frozen_96_bytes() {
        assert_eq!(KERNARG_BYTES, 96);
        let g = Geom {
            xrs: 2560,
            ys: 640,
            x_src_rows: 1,
        };
        let a = kernargs_of(E8pKind::GateUp, 1, 10, [1, 2, 3, 4, 5, 6, 7], g).to_bytes();
        assert_eq!(a.len(), 96);
        for i in 0..7usize {
            assert_eq!(
                u64::from_le_bytes(a[i * 8..i * 8 + 8].try_into().unwrap()),
                i as u64 + 1
            );
        }
        let w = |o: usize| u32::from_le_bytes(a[o..o + 4].try_into().unwrap());
        assert_eq!(
            [w(56), w(60), w(64), w(68), w(72), w(76), w(80), w(84)],
            [1280, 2560, 10, 1, 10, 1, 2560, 640]
        );
        assert_eq!(w(88), N_EXPERTS as u32, "n_experts is the table extent");
        assert_eq!(w(92), 0, "reserved word");
    }

    #[test]
    fn bad_id_case_keeps_every_bad_expert_slot_on_the_canary() {
        let mut cache = ExpertCache::new();
        let case = build_synthetic("c1_bad_ids", &mut cache, 0).unwrap();
        assert!(
            !case.desc_in_contract(),
            "ids at or past the table extent are outside the planner contract"
        );
        assert_eq!(case.desc.ids[1..5], BAD_IDS);
        for kind in E8pKind::ALL {
            let e = expected(&case, kind, false).unwrap();
            assert_eq!(e.written_slots, 2, "only the two valid experts store");
            let ys = case.geom[kind.index()].ys;
            for slot in 1..5usize {
                assert!(
                    e.y0[slot * ys..slot * ys + kind.out_rows()]
                        .iter()
                        .all(|v| v.to_bits() == SENTINEL_BITS),
                    "slot {slot}"
                );
            }
        }
    }
}
