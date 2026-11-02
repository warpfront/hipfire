// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `e8p_nan_containment`: the expert-slab E8P (and, on gfx1100, hot MQ4) K1/K2 consumers of the
//! MoE non-finite router guard, old unguarded images vs the new guarded images, on gfx1201, gfx1100 and gfx1151.
//!
//! A child module of `moe_nan_guard_regress.rs` (it reuses that driver's `Ctx`, old-image loader,
//! fixtures, router runners and comparators). It runs ONLY on an arch whose slab matrix admits
//! E8P (gfx1201, gfx1100, gfx1151); there is no skip on those arches.
//!
//! # What is proved
//!
//! 1. **Finite old == new, bitwise.** The old image (rc4b, or the pre-rc5 gfx1151 image; 88-byte K1 / 60-byte K2 kernarg) and the new
//!    guarded production launcher (96-byte K1 / 64-byte K2 kernarg) run on the same valid finite
//!    inputs and the same routing: E8P K1 gate/up + down at 1/3/8 lanes, E8P K2 gate/up + down at
//!    7 and 33 prefill rows (gate output feeds the down A4 sidecar through the real producers);
//!    on gfx1100 the hot MQ4 K1/K2 likewise (their old images are the rc4b `mq4only` / `iu4_sym`
//!    images). The K1 old image reads the production planner's union; only the K1 image differs.
//! 2. **Raw invalid ids skip, store nothing.** Ids `-1`, `n_experts`, `INT_MAX`, `INT_MIN` planted
//!    into the K1 union (and, mixed in one launch, all four) and into the K2 tile table launch ONLY
//!    the NEW images. The native PM skip does NOT zero-fill: every output region of an invalid
//!    expert/tile is pre-filled with a FINITE canary (`0xA5A5A5A5` words / `0xA5` bytes) and must
//!    still be exactly the canary afterwards; every valid region is bitwise the old image's.
//!    The old images are never launched with an invalid id. A launch or sync error on a bad id
//!    counts as a fault mismatch.
//! 3. **The guard has teeth.** A shortened declared `n_experts` (256) over the same valid physical
//!    tables/unions MUST change the finite comparison (every expert id >= 256 skips): K1 through a
//!    raw launch with `n_experts=256`, K2 through the production launcher over a 256-entry view of
//!    the pointer table (the launcher declares the table extent). The skipped regions must be
//!    exactly canary and the rest bitwise unchanged. The declared count is the full table extent,
//!    never the number of active experts.
//! 4. **Router producer + consumer chain.** All-NaN, +Inf and -Inf router rows (the real router
//!    kernels, incl. the fast one on gfx1201 and gfx1151) fail closed: every id -1, every weight +0, the typed
//!    fault word names `(layer, global row)` exactly (single failing row) or one failing row
//!    (several); finite rows are bitwise the pinned beta router's. The router's device tensors
//!    feed the production planner + K1 (decode) and the slab scatter + A4 producers + K2
//!    (prefill); failed-row slots stay canary, valid-slot rows equal the old image's chain over
//!    the clean-twin routing. The pinned beta router fed an all-NaN row does NOT fail closed (the
//!    router's teeth control).
//!
//! # Publication
//!
//! `<out-dir>/e8p_nan_containment.<arch>.json` (schema `c24-e8p-nan-producer-contract-v1`) and
//! `<out-dir>/e8p_nan_containment.<arch>.f32le`: the real GPU output words of every valid finite
//! comparison run (K1 f32 rows verbatim; K2 BF16 rows of the routed slots widened exactly to f32),
//! checked finite before the file is written. Stale outputs are removed at case start; the file is
//! written only on PASS; the JSON is written on PASS and FAIL.
//!
//! # Old inputs (`--old-dir`, flat)
//!
//! rc4b images: `e8p_i4_moe_gemv_pm_<arch>.hxaco`, `qwen4_moe_iu4_sym_e8p_pm_<arch>.hxaco`
//! (+ gfx1100 `mq_i4_moe_gemv_mq4only_pm_gfx1100.hxaco`, `qwen4_moe_iu4_sym_pm_gfx1100.hxaco`);
//! pinned beta sources `moe_router_softmax_top10_f32.hip` (+ `_fast.hip` on gfx1201).
//!
//! gfx1151 has no rc4b image at the pinned beta ref (its E8P kernels first exist in the pre-rc5
//! gfx1151 port, commit `11411610b`, whose images are the unguarded 88-byte K1 / 60-byte K2 ABI:
//! `e8p_i4_moe_gemv_pm_gfx1151.hxaco` sha256 `4cd87a1e…`, `qwen4_moe_iu4_sym_e8p_pm_gfx1151.hxaco`
//! sha256 `4312e5bc…`); those two files, taken from that commit, are the gfx1151 old baseline in
//! `--old-dir` (the current tree's same-named files are the new bounded images and are NOT the
//! baseline). gfx1151 compares E8P only: there is no old gfx1151 MQ4 K2 image, so no hot-MQ4
//! phase there (the hot MQ4 K1 is compared against beta's image by `pm_k1_case`).

use super::*;
use hipfire_runtime::e8p_i4_reference as e8p;
use rdna_compute::scratch::Int4MmqPrepared;
use rdna_compute::slab_admission::{SLAB_CODEC_E8P, SLAB_CODEC_MQ4};
use rdna_compute::slab_moe;
use serde_json::{json, Value};

pub(super) const CASE_NAME: &str = "e8p_nan_containment";
pub(super) const SCHEMA: &str = "c24-e8p-nan-producer-contract-v1";

/// Finite canary of every raw-consumer output region (f32 word / byte).
const CANARY_WORD: u32 = 0xA5A5_A5A5;
const CANARY_BYTE: u8 = 0xA5;
const LANES: [usize; 3] = [1, 3, 8];
const K2_ROWS: [usize; 2] = [7, 33];
/// Prefill rows of the router -> scatter -> K2 chain.
const CHAIN_ROWS: usize = 9;
/// Declared `n_experts` of the teeth controls (half the 512-entry table).
const SHORT_N_EXPERTS: usize = 256;
const ROUTER_LAYER: usize = 5;
const ROUTER_ROW_BASE: usize = 64;
/// Router kinds of the E8P chains: all-NaN, +Inf, -Inf, all +Inf rows.
const ROUTER_BADS: [Bad; 4] = [Bad::AllNan, Bad::PlusInf, Bad::MinusInf, Bad::AllPlusInf];

const GATE_M: usize = slab_moe::SLAB_GATE_UP_M;
const GATE_K: usize = slab_moe::SLAB_GATE_UP_K;
const DOWN_MM: usize = slab_moe::SLAB_DOWN_M;
const DOWN_KK: usize = slab_moe::SLAB_DOWN_K;

/// `(m, k, x_row_div)` of the gate/up or down projection.
fn proj_dims(down: bool) -> (usize, usize, usize) {
    if down {
        (DOWN_MM, DOWN_KK, 1)
    } else {
        (GATE_M, GATE_K, K)
    }
}

/// Bytes of one planar E8P expert blob `[M*K/8 u16 words][M*K/128 f16 scales]`.
const fn e8p_blob_bytes(m: usize, k: usize) -> usize {
    m * k / 4 + m * k / 64
}
/// Bytes of one MQ4G256V2 gate/up expert and one MQ4G128V2 down expert.
const fn mq4_gate_bytes(m: usize, k: usize) -> usize {
    m * (k / 256) * 136
}
const fn mq4_down_bytes(m: usize, k: usize) -> usize {
    m * (k / 128) * 68
}

fn sha256_hex(d: &[u8]) -> String {
    Sha256::digest(d)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

// ───────────────────────────────────────────────────────────────────────────────
// Counters, report, publication
// ───────────────────────────────────────────────────────────────────────────────
#[derive(Default)]
struct Counters {
    finite_exact_compares: u64,
    finite_exact_compare_bytes: u64,
    finite_compare_mismatches: u64,
    raw_bad_id_cases: u64,
    all_failed_cases: u64,
    raw_bad_id_canary_regions: u64,
    raw_bad_id_valid_regions: u64,
    raw_bad_id_store_mismatches: u64,
    raw_bad_id_valid_mismatches: u64,
    raw_bad_id_fault_mismatches: u64,
    positive_controls: u64,
    teeth_controls: u64,
    teeth_skipped_regions: u64,
    router_launches: u64,
    router_failed_rows: u64,
    router_finite_rows: u64,
    router_contract_mismatches: u64,
    router_fault_mismatches: u64,
    router_old_teeth_controls: u64,
    chain_runs: u64,
    chain_valid_slot_compares: u64,
    chain_failed_slot_checks: u64,
    chain_mismatches: u64,
    output_elements: u64,
    output_nonfinite: u64,
}

impl Counters {
    fn router_containment(&self) -> bool {
        self.router_launches > 0
            && self.router_failed_rows > 0
            && self.router_finite_rows > 0
            && self.router_contract_mismatches == 0
            && self.router_fault_mismatches == 0
            && self.router_old_teeth_controls > 0
            && self.chain_runs > 0
            && self.chain_failed_slot_checks > 0
            && self.chain_valid_slot_compares > 0
            && self.chain_mismatches == 0
    }

    /// Every acceptance counter of the contract, independent of the check list.
    fn accepted(&self) -> bool {
        self.finite_exact_compares > 0
            && self.finite_compare_mismatches == 0
            && self.raw_bad_id_cases > 0
            && self.raw_bad_id_canary_regions > 0
            && self.raw_bad_id_valid_regions > 0
            && self.raw_bad_id_store_mismatches == 0
            && self.raw_bad_id_valid_mismatches == 0
            && self.raw_bad_id_fault_mismatches == 0
            && self.positive_controls > 0
            && self.teeth_controls > 0
            && self.router_containment()
            && self.output_elements > 0
            && self.output_nonfinite == 0
    }

    fn to_json(&self) -> Value {
        json!({
            "finite_exact_compares": self.finite_exact_compares,
            "finite_exact_compare_bytes": self.finite_exact_compare_bytes,
            "finite_compare_mismatches": self.finite_compare_mismatches,
            "raw_bad_id_cases": self.raw_bad_id_cases,
            "all_failed_cases": self.all_failed_cases,
            "raw_bad_id_canary_regions": self.raw_bad_id_canary_regions,
            "raw_bad_id_valid_regions": self.raw_bad_id_valid_regions,
            "raw_bad_id_store_mismatches": self.raw_bad_id_store_mismatches,
            "raw_bad_id_valid_mismatches": self.raw_bad_id_valid_mismatches,
            "raw_bad_id_fault_mismatches": self.raw_bad_id_fault_mismatches,
            "positive_controls": self.positive_controls,
            "teeth_controls": self.teeth_controls,
            "teeth_skipped_regions": self.teeth_skipped_regions,
            "router_launches": self.router_launches,
            "router_failed_rows": self.router_failed_rows,
            "router_finite_rows": self.router_finite_rows,
            "router_contract_mismatches": self.router_contract_mismatches,
            "router_fault_mismatches": self.router_fault_mismatches,
            "router_old_teeth_controls": self.router_old_teeth_controls,
            "router_containment": self.router_containment(),
            "chain_runs": self.chain_runs,
            "chain_valid_slot_compares": self.chain_valid_slot_compares,
            "chain_failed_slot_checks": self.chain_failed_slot_checks,
            "chain_mismatches": self.chain_mismatches,
            "output_elements": self.output_elements,
            "output_nonfinite": self.output_nonfinite,
        })
    }
}

struct Segment {
    label: String,
    source: &'static str,
    offset: usize,
    elements: usize,
    sha256: String,
}

#[derive(Default)]
struct Report {
    c: Counters,
    results: Vec<R<String>>,
    /// The published finite f32 output (concatenated segments).
    out: Vec<f32>,
    segments: Vec<Segment>,
    images: Vec<Value>,
    sources: Vec<Value>,
    phases: Vec<String>,
}

/// Why a region check failed: an invalid region was written, a valid one differs.
enum RegionFault {
    Stored(String),
    Valid(String),
}

impl RegionFault {
    fn msg(&self) -> &str {
        match self {
            Self::Stored(m) | Self::Valid(m) => m,
        }
    }
}

/// Every K1 slot row of `got`: a `skipped` slot must be entirely the canary, every other slot
/// bitwise `want`'s. Returns `(canary regions, valid regions)` checked.
fn k1_regions(
    label: &str,
    got: &K1Out,
    want: &K1Out,
    skipped: &[bool],
    ys: usize,
    canary: u32,
) -> Result<(u64, u64), RegionFault> {
    let (mut canary_regions, mut valid_regions) = (0u64, 0u64);
    for (g, w, name) in [(&got.y0, &want.y0, "y0"), (&got.y1, &want.y1, "y1")] {
        if g.is_empty() && w.is_empty() {
            continue;
        }
        if g.len() != w.len() || g.len() != skipped.len() * ys {
            return Err(RegionFault::Valid(format!(
                "{label} {name}: output words {} vs {} vs {} slots x {ys}",
                g.len(),
                w.len(),
                skipped.len()
            )));
        }
        for (slot, &skip) in skipped.iter().enumerate() {
            let (a, b) = (
                &g[slot * ys..(slot + 1) * ys],
                &w[slot * ys..(slot + 1) * ys],
            );
            if skip {
                if a.iter().any(|&v| v != canary) {
                    return Err(RegionFault::Stored(format!(
                        "{label} {name}: skipped slot {slot} was written"
                    )));
                }
                canary_regions += 1;
            } else {
                if a != b {
                    return Err(RegionFault::Valid(format!(
                        "{label} {name}: valid slot {slot} differs"
                    )));
                }
                valid_regions += 1;
            }
        }
    }
    Ok((canary_regions, valid_regions))
}

/// Every `tile_bytes` tile of `got`: a `skipped` tile must be entirely `CANARY_BYTE`, every other
/// tile bitwise `want`'s. Returns `(canary tiles, valid tiles)`.
fn k2_tile_regions(
    label: &str,
    got: &[u8],
    want: &[u8],
    skipped: &[bool],
    tile_bytes: usize,
) -> Result<(u64, u64), RegionFault> {
    if got.len() != want.len() || got.len() != skipped.len() * tile_bytes {
        return Err(RegionFault::Valid(format!(
            "{label}: output bytes {} vs {} vs {} tiles x {tile_bytes}",
            got.len(),
            want.len(),
            skipped.len()
        )));
    }
    let (mut canary_tiles, mut valid_tiles) = (0u64, 0u64);
    for (tile, &skip) in skipped.iter().enumerate() {
        let (a, b) = (
            &got[tile * tile_bytes..(tile + 1) * tile_bytes],
            &want[tile * tile_bytes..(tile + 1) * tile_bytes],
        );
        if skip {
            if a.iter().any(|&v| v != CANARY_BYTE) {
                return Err(RegionFault::Stored(format!(
                    "{label}: skipped tile {tile} was written"
                )));
            }
            canary_tiles += 1;
        } else {
            if a != b {
                return Err(RegionFault::Valid(format!(
                    "{label}: valid tile {tile} differs"
                )));
            }
            valid_tiles += 1;
        }
    }
    Ok((canary_tiles, valid_tiles))
}

impl Report {
    fn cmp_words(&mut self, label: &str, got: &[u32], want: &[u32]) {
        self.c.finite_exact_compares += 1;
        self.c.finite_exact_compare_bytes += 4 * got.len() as u64;
        let r = if got.is_empty() {
            Err(format!("{label}: vacuous compare"))
        } else {
            exact_ints(label, got, want)
        };
        if r.is_err() {
            self.c.finite_compare_mismatches += 1;
        }
        self.results.push(r);
    }

    fn cmp_bytes(&mut self, label: &str, got: &[u8], want: &[u8]) {
        self.c.finite_exact_compares += 1;
        self.c.finite_exact_compare_bytes += got.len() as u64;
        let r = if got.is_empty() {
            Err(format!("{label}: vacuous compare"))
        } else {
            exact_ints(label, got, want)
        };
        if r.is_err() {
            self.c.finite_compare_mismatches += 1;
        }
        self.results.push(r);
    }

    fn image(&mut self, cx: &Ctx, rel: &str) -> R<()> {
        let bytes = cx.old_blob(rel)?;
        self.images.push(json!({
            "file": rel,
            "bytes": bytes.len(),
            "sha256": sha256_hex(&bytes),
            "from": if cx.old_dir.is_some() { "--old-dir" } else { "git old-ref" },
        }));
        Ok(())
    }

    fn source(&mut self, cx: &Ctx, old: &OldFile) -> R<()> {
        let text = cx.old_source(old.file)?;
        self.sources
            .push(json!({"file": old.file, "sha256": sha256_hex(text.as_bytes())}));
        Ok(())
    }

    /// A bad-id launch that errored (device fault, launch refusal, sync error).
    fn bad_launch_failed(&mut self, label: &str, err: String) {
        self.c.raw_bad_id_cases += 1;
        self.c.raw_bad_id_fault_mismatches += 1;
        self.results.push(Err(format!(
            "{label}: launch/sync failed on an invalid id: {err}"
        )));
    }

    /// Record a bad-id region check: skipped regions canary, valid regions bitwise. `all_failed`
    /// is true ONLY when the caller's mask skips every region (a fully failed-closed router row at
    /// one lane): then the check is `canary > 0 && valid == 0` and counts as an all-failed case;
    /// otherwise both counts must be nonzero (raw-id and K2 cases are never all-failed).
    fn bad_regions(&mut self, label: &str, r: Result<(u64, u64), RegionFault>, all_failed: bool) {
        self.c.raw_bad_id_cases += 1;
        match r {
            Ok((canary, valid)) => {
                self.c.raw_bad_id_canary_regions += canary;
                self.c.raw_bad_id_valid_regions += valid;
                if all_failed && canary > 0 && valid == 0 {
                    self.c.all_failed_cases += 1;
                    self.c.positive_controls += 1;
                    self.results.push(Ok(format!(
                        "{label}: every region invalid, {canary} stayed canary, nothing stored"
                    )));
                } else if !all_failed && canary > 0 && valid > 0 {
                    self.c.positive_controls += 1;
                    self.results.push(Ok(format!("{label}: {canary} invalid regions stayed canary, {valid} valid regions == old")));
                } else {
                    self.results.push(Err(format!(
                        "{label}: vacuous ({canary} canary, {valid} valid regions, all_failed={all_failed})"
                    )));
                }
            }
            Err(fault) => {
                match fault {
                    RegionFault::Stored(_) => self.c.raw_bad_id_store_mismatches += 1,
                    RegionFault::Valid(_) => self.c.raw_bad_id_valid_mismatches += 1,
                }
                self.results.push(Err(fault.msg().to_owned()));
            }
        }
    }

    /// Record a shortened-`n_experts` control: it MUST skip (canary) at least one region the full
    /// declared count computes, and nothing else may change.
    fn teeth_regions(&mut self, label: &str, r: Result<(u64, u64), RegionFault>) {
        match r {
            Ok((skipped, valid)) if skipped > 0 && valid > 0 => {
                self.c.teeth_controls += 1;
                self.c.teeth_skipped_regions += skipped;
                self.results.push(Ok(format!(
                    "{label}: shortened n_experts={SHORT_N_EXPERTS} skipped {skipped} regions (canary) the full count computes, {valid} unchanged"
                )));
            }
            Ok((skipped, valid)) => self.results.push(Err(format!(
                "{label}: shortened n_experts did not discriminate ({skipped} skipped, {valid} unchanged regions): the guard has no teeth"
            ))),
            Err(fault) => self.results.push(Err(format!("{label} (teeth): {}", fault.msg()))),
        }
    }

    fn publish_f32(&mut self, label: &str, source: &'static str, vals: impl Iterator<Item = f32>) {
        let offset = self.out.len();
        let mut h = Sha256::new();
        for v in vals {
            if !v.is_finite() {
                self.c.output_nonfinite += 1;
            }
            h.update(v.to_le_bytes());
            self.out.push(v);
        }
        let elements = self.out.len() - offset;
        self.c.output_elements += elements as u64;
        let sha256 = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
        self.segments.push(Segment {
            label: label.to_owned(),
            source,
            offset,
            elements,
            sha256,
        });
    }

    /// Publish the BF16 rows of `bytes` at `row_bytes` stride for every routed slot position.
    fn publish_bf16_rows(
        &mut self,
        label: &str,
        bytes: &[u8],
        row_bytes: usize,
        positions: &[usize],
    ) -> R<()> {
        let mut vals = Vec::with_capacity(positions.len() * row_bytes / 2);
        for &pos in positions {
            let row = bytes
                .get(pos * row_bytes..(pos + 1) * row_bytes)
                .ok_or_else(|| format!("{label}: row position {pos} outside the output"))?;
            if row.iter().all(|&b| b == CANARY_BYTE) {
                return Err(format!(
                    "{label}: routed row {pos} is still the canary (never written)"
                ));
            }
            vals.extend(
                row.chunks_exact(2)
                    .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]]))),
            );
        }
        self.publish_f32(label, "bf16_widened", vals.into_iter());
        self.c.positive_controls += 1;
        Ok(())
    }
}

// ───────────────────────────────────────────────────────────────────────────────
// Fixtures
// ───────────────────────────────────────────────────────────────────────────────
/// Planar E8P expert blob (random words / `pack_word` parity words; guard-passing normal scales
/// of magnitude <= 0.5 so every sum and the SwiGLU/A4 chain stay far inside f32/BF16 range).
fn e8p_blob(rng: &mut Rng, m: usize, k: usize, mode: usize) -> Vec<u8> {
    let (nw, ns) = (m * k / 8, m * k / 128);
    let mut out = Vec::with_capacity(e8p_blob_bytes(m, k));
    for i in 0..nw {
        let w: u16 = if mode % 2 == 0 {
            rng.next_u64() as u16
        } else {
            e8p::pack_word(
                ((i * 7 + mode) % 256) as u8,
                ((i * 37 + (i >> 8)) & 0x7f) as u8,
                i & 1 == 1,
            )
        };
        out.extend_from_slice(&w.to_le_bytes());
    }
    for _ in 0..ns {
        let d: u16 = ((8 + rng.below(6) as u16) << 10) | (rng.next_u64() as u16 & 0x3ff);
        debug_assert!(e8p::writer_guard(d).is_ok());
        out.extend_from_slice(&d.to_le_bytes());
    }
    out
}

fn build_e8p_experts(cx: &mut Ctx, rng: &mut Rng, m: usize, k: usize) -> R<ExpertTable> {
    let mut real = Vec::with_capacity(N_REAL);
    for j in 0..N_REAL {
        let bytes = e8p_blob(rng, m, k, j);
        real.push(up_bytes(&mut cx.gpu, &bytes)?);
    }
    ExpertTable::build(&mut cx.gpu, real)
}

struct Tables {
    gate: ExpertTable,
    down: ExpertTable,
}

impl Tables {
    fn build(cx: &mut Ctx, rng: &mut Rng, codec: u8) -> R<Self> {
        match codec {
            SLAB_CODEC_E8P => Ok(Self {
                gate: build_e8p_experts(cx, rng, GATE_M, GATE_K)?,
                down: build_e8p_experts(cx, rng, DOWN_MM, DOWN_KK)?,
            }),
            SLAB_CODEC_MQ4 => Ok(Self {
                gate: build_experts_with(cx, rng, mq4_gate_expert, GATE_M, GATE_K)?,
                down: build_experts_with(cx, rng, mq4_down_expert, DOWN_MM, DOWN_KK)?,
            }),
            other => Err(format!("no fixture for codec {other}")),
        }
    }

    fn free(self, cx: &mut Ctx) -> R<()> {
        self.gate.free(&mut cx.gpu)?;
        self.down.free(&mut cx.gpu)
    }
}

/// The first `n` entries of a pointer table: a borrowed view whose extent is the declared
/// `n_experts` of the production K2 launchers. Never freed (the owning table is).
fn short_view(table: &GpuTensor, n: usize) -> GpuTensor {
    GpuTensor {
        buf: table.buf.byte_view(0, n * 8),
        shape: vec![n * 2],
        dtype: rdna_compute::DType::F32,
    }
}

/// One slab codec the case exercises on this arch.
struct Phase {
    codec: u8,
    label: &'static str,
    k1: K1Target,
    k1_old_rel: String,
    k2_old_rel: String,
    k2_syms: [&'static str; 2],
}

fn k2_old_stem(arch: &str, codec: u8) -> R<&'static str> {
    match (arch, codec) {
        ("gfx1201", SLAB_CODEC_E8P) => Ok("qwen4_moe_iu4_sym_e8p_pm_gfx1201"),
        ("gfx1100", SLAB_CODEC_E8P) => Ok("qwen4_moe_iu4_sym_e8p_pm_gfx1100"),
        ("gfx1151", SLAB_CODEC_E8P) => Ok("qwen4_moe_iu4_sym_e8p_pm_gfx1151"),
        ("gfx1100", SLAB_CODEC_MQ4) => Ok("qwen4_moe_iu4_sym_pm_gfx1100"),
        _ => Err(format!("no old K2 image for codec {codec} on {arch}")),
    }
}

fn phase(cx: &Ctx, codec: u8) -> R<Phase> {
    let arch = cx.arch.as_str();
    let slab =
        slab_moe::slab_arch_kernels(arch).ok_or_else(|| format!("{arch} has no slab table"))?;
    let (k1, label) = match codec {
        SLAB_CODEC_E8P => {
            let cold = slab_moe::slab_cold_kernels(arch, SLAB_CODEC_E8P)
                .ok_or_else(|| format!("{arch}: E8P is not served by the slab"))?;
            (
                (
                    K1Target {
                        bucket: 0,
                        cold_codec: SLAB_CODEC_E8P,
                        symbols: cold.k1.symbols,
                        canary: CANARY_WORD,
                    },
                    cold.k1.module,
                ),
                "E8P",
            )
        }
        SLAB_CODEC_MQ4 => (
            (
                K1Target {
                    bucket: 1,
                    cold_codec: SLAB_CODEC_MQ4,
                    symbols: slab.mq4_k1.symbols,
                    canary: CANARY_WORD,
                },
                slab.mq4_k1.module,
            ),
            "MQ4-hot",
        ),
        other => return Err(format!("no phase for codec {other}")),
    };
    let k2_syms = cx
        .gpu
        .qwen4_moe_sym_slab_gemm_symbols(codec)
        .ok_or_else(|| format!("{arch}: no K2 slab entry for codec {codec}"))?;
    Ok(Phase {
        codec,
        label,
        k1: k1.0,
        k1_old_rel: format!("kernels/{}.hxaco", k1.1),
        k2_old_rel: format!("kernels/{}.hxaco", k2_old_stem(arch, codec)?),
        k2_syms,
    })
}

/// A uniform valid `topk` of `lanes` lanes with experts on both sides of `SHORT_N_EXPERTS`, so the
/// shortened-count teeth control skips some regions and keeps others.
fn straddling_topk(rng: &mut Rng, lanes: usize) -> R<Vec<i32>> {
    for _ in 0..256 {
        let topk = make_topk(rng, lanes, Spread::Uniform);
        let low = topk.iter().any(|&id| (id as usize) < SHORT_N_EXPERTS);
        let high = topk.iter().any(|&id| (id as usize) >= SHORT_N_EXPERTS);
        if low && high {
            return Ok(topk);
        }
    }
    Err(format!(
        "no {lanes}-lane topk straddles n_experts={SHORT_N_EXPERTS}"
    ))
}

// ───────────────────────────────────────────────────────────────────────────────
// K1
// ───────────────────────────────────────────────────────────────────────────────
fn k1_phase(cx: &mut Ctx, rep: &mut Report, ph: &Phase, tabs: &Tables, rng: &mut Rng) -> R<()> {
    let (_module, fns) = cx.load_old_image(&ph.k1_old_rel, &ph.k1.symbols)?;
    rep.image(cx, &ph.k1_old_rel)?;
    let canary = ph.k1.canary;
    for &lanes in &LANES {
        let s = K * lanes;
        let topk = straddling_topk(rng, lanes)?;
        for is_down in [false, true] {
            let (_, k, _, srows, ys) = k1_dims(is_down, lanes);
            let wtab = if is_down { &tabs.down } else { &tabs.gate };
            let f = &fns[usize::from(is_down)];
            let x = rng.vec(srows * k, -2.0, 2.0);
            let kind = if is_down { "down" } else { "gate_up" };
            let tag = format!("{} K1 {kind} lanes={lanes}", ph.label);
            // 1. finite old image == new guarded launcher, bitwise.
            let old = k1_run(
                cx,
                K1Path::OldImage(f),
                ph.k1,
                is_down,
                &topk,
                lanes,
                wtab,
                &x,
            )?;
            let new = k1_run(cx, K1Path::New, ph.k1, is_down, &topk, lanes, wtab, &x)?;
            rep.cmp_words(&format!("{tag} y0 new-vs-old"), &new.y0, &old.y0);
            if !is_down {
                rep.cmp_words(&format!("{tag} y1 new-vs-old"), &new.y1, &old.y1);
            }
            if (0..s).any(|slot| rows_untouched(&new.y0, ys, slot, canary)) {
                rep.results.push(Err(format!(
                    "{tag}: a routed slot row was never written (still the canary)"
                )));
            } else {
                rep.c.positive_controls += 1;
            }
            rep.publish_f32(&tag, "f32", new.y0.iter().map(|&b| f32::from_bits(b)));
            if !is_down {
                rep.publish_f32(
                    &format!("{tag} up"),
                    "f32",
                    new.y1.iter().map(|&b| f32::from_bits(b)),
                );
            }
            // 2. invalid ids through the production planner (unrouted slots): canary, valid == old.
            for phase_id in 0..4 {
                let (bad_ids, bad) = poison_topk(&topk, phase_id);
                let label = format!("{tag} planner phase={phase_id}");
                match k1_run(cx, K1Path::New, ph.k1, is_down, &bad_ids, lanes, wtab, &x) {
                    Ok(got) => {
                        let all_failed = bad.iter().all(|&b| b);
                        rep.bad_regions(
                            &label,
                            k1_regions(&label, &got, &old, &bad, ys, canary),
                            all_failed,
                        )
                    }
                    Err(err) => rep.bad_launch_failed(&label, err),
                }
            }
            // 3. raw invalid ids planted in the union, NEW image only (never the old image).
            let singles = BAD_IDS.iter().map(|&id| vec![(0usize, id)]);
            let mixed = std::iter::once(
                BAD_IDS
                    .iter()
                    .enumerate()
                    .map(|(j, &id)| (j, id))
                    .collect::<Vec<_>>(),
            );
            for plant in singles.chain(mixed) {
                let union = prod_union(cx, &topk, lanes, ph.k1.bucket)?;
                let mut words = down_u32(&cx.gpu, &union)?;
                let (ids_w, offs_w, slots_w) = k1_union_words(s, ph.k1.bucket);
                let mut planted = vec![false; s];
                for &(j, id) in &plant {
                    let (lo, hi) = (words[offs_w + j] as usize, words[offs_w + j + 1] as usize);
                    for p in lo..hi {
                        planted[words[slots_w + p] as usize] = true;
                    }
                    words[ids_w + j] = id as u32;
                }
                let t_union = up_u32(&mut cx.gpu, &words)?;
                let label = format!(
                    "{tag} raw union ids {:?}",
                    plant.iter().map(|p| p.1).collect::<Vec<_>>()
                );
                match k1_run(
                    cx,
                    K1Path::Raw(&t_union, E as u32),
                    ph.k1,
                    is_down,
                    &topk,
                    lanes,
                    wtab,
                    &x,
                ) {
                    Ok(got) => rep.bad_regions(
                        &label,
                        k1_regions(&label, &got, &old, &planted, ys, canary),
                        false,
                    ),
                    Err(err) => rep.bad_launch_failed(&label, err),
                }
                free_all(&mut cx.gpu, vec![union, t_union])?;
            }
            // 4. teeth: a shortened declared n_experts over the same valid union MUST change the result.
            let union = prod_union(cx, &topk, lanes, ph.k1.bucket)?;
            let skipped: Vec<bool> = topk
                .iter()
                .map(|&id| id as usize >= SHORT_N_EXPERTS)
                .collect();
            let got = k1_run(
                cx,
                K1Path::Raw(&union, SHORT_N_EXPERTS as u32),
                ph.k1,
                is_down,
                &topk,
                lanes,
                wtab,
                &x,
            )?;
            let label = format!("{tag} n_experts={SHORT_N_EXPERTS}");
            rep.teeth_regions(&label, k1_regions(&label, &got, &old, &skipped, ys, canary));
            free_all(&mut cx.gpu, vec![union])?;
        }
    }
    Ok(())
}

/// Router (device tensors) -> production planner -> production K1 vs the old image over the
/// clean-twin routing: failed-row slots stay canary, valid slots bitwise the old image's.
fn k1_chain(cx: &mut Ctx, rep: &mut Report, ph: &Phase, tabs: &Tables, rng: &mut Rng) -> R<()> {
    let (_module, fns) = cx.load_old_image(&ph.k1_old_rel, &ph.k1.symbols)?;
    let canary = ph.k1.canary;
    for &lanes in &[3usize, 8] {
        for (li, kinds) in chain_layouts(lanes).iter().enumerate() {
            let (logits, clean, failing) = layout_logits(rng, kinds, li);
            let (old_ids, old_w) = run_router_old(cx, &clean, lanes, false, true, true)?;
            let dev = router_device(cx, &logits, lanes, false, ROUTER_LAYER, ROUTER_ROW_BASE)?;
            let ids = down_i32(&cx.gpu, &dev.ids)?;
            let w = bits(&down_f32(&cx.gpu, &dev.w)?);
            let out = RouterOut {
                ids,
                w,
                fault: dev.fault,
            };
            router_check(
                rep,
                &format!("K1 chain lanes={lanes} layout={li}"),
                &out,
                &failing,
                &old_ids,
                &old_w,
                ROUTER_LAYER,
                ROUTER_ROW_BASE,
            );
            let skipped: Vec<bool> = (0..lanes * K).map(|slot| failing[slot / K]).collect();
            for is_down in [false, true] {
                let (_, k, _, srows, ys) = k1_dims(is_down, lanes);
                let wtab = if is_down { &tabs.down } else { &tabs.gate };
                let x = rng.vec(srows * k, -2.0, 2.0);
                let kind = if is_down { "down" } else { "gate_up" };
                let label = format!("{} K1 chain {kind} lanes={lanes} layout={li}", ph.label);
                let old = k1_run(
                    cx,
                    K1Path::OldImage(&fns[usize::from(is_down)]),
                    ph.k1,
                    is_down,
                    &old_ids,
                    lanes,
                    wtab,
                    &x,
                )?;
                let got = k1_run(
                    cx,
                    K1Path::NewDev(&dev.ids),
                    ph.k1,
                    is_down,
                    &out.ids,
                    lanes,
                    wtab,
                    &x,
                )?;
                rep.chain_regions(&label, k1_regions(&label, &got, &old, &skipped, ys, canary));
            }
            free_all(&mut cx.gpu, vec![dev.ids, dev.w])?;
        }
    }
    Ok(())
}

// ───────────────────────────────────────────────────────────────────────────────
// K2
// ───────────────────────────────────────────────────────────────────────────────
fn k2_gemm(
    cx: &mut Ctx,
    codec: u8,
    down: bool,
    imp: Option<&hip_bridge::Function>,
    table: &GpuTensor,
    tiles: &GpuTensor,
    sorted: &GpuTensor,
    xq: &Int4MmqPrepared,
    y: &GpuTensor,
    gr: usize,
    src_rows: usize,
) -> R<()> {
    let (m, k, div) = proj_dims(down);
    if let Some(f) = imp {
        let (generation, live) = if down {
            cx.gpu.scratch.qwen4_moe_down_i4_live()
        } else {
            cx.gpu.scratch.int4_mmq_live()
        };
        let xp = xq.checked_ptr(generation, live, k, src_rows).map_err(e)?;
        let mut b = KernargBlob::new();
        b.push_ptr(ptr(table));
        b.push_ptr(ptr(tiles));
        b.push_ptr(ptr(sorted));
        b.push_ptr(xp as *const c_void);
        b.push_ptr(ptr(y));
        for v in [m, k, div, gr, src_rows] {
            b.push_i32(v as i32);
        }
        // NT1 grid (m / 64, grouped_rows / 16); block 64 (gate/up) or 128 (down): the OLD 60-byte ABI.
        return cx.launch_raw(
            f,
            [(m / 64) as u32, (gr / 16) as u32, 1],
            [if down { 128 } else { 64 }, 1, 1],
            b,
        );
    }
    let launched = match (codec, down) {
        (SLAB_CODEC_E8P, false) => cx.gpu.gemm_qwen4_moe_e8p_gate_up_silu_iu4_sym(
            table, tiles, sorted, xq, y, m, k, div, gr, src_rows,
        ),
        (SLAB_CODEC_E8P, true) => cx
            .gpu
            .gemm_qwen4_moe_e8p_down_iu4_sym(table, tiles, sorted, xq, y, m, k, div, gr, src_rows),
        (SLAB_CODEC_MQ4, false) => cx.gpu.gemm_qwen4_moe_gate_up_silu_iu4_sym(
            table, tiles, sorted, xq, y, m, k, div, gr, src_rows, false,
        ),
        (SLAB_CODEC_MQ4, true) => cx.gpu.gemm_qwen4_moe_down_iu4_sym(
            table, tiles, sorted, xq, y, m, k, div, gr, src_rows, false,
        ),
        (other, _) => return Err(format!("no K2 launcher for codec {other}")),
    };
    launched.map_err(e)
}

/// One K2 launch over host `tiles` into a canary-filled BF16 `y`; returns the downloaded bytes
/// and the `y` tensor (the caller frees it).
fn k2_run(
    cx: &mut Ctx,
    codec: u8,
    down: bool,
    imp: Option<&hip_bridge::Function>,
    table: &GpuTensor,
    tiles_host: &[i32],
    sorted: &GpuTensor,
    xq: &Int4MmqPrepared,
    gr: usize,
    src_rows: usize,
) -> R<(Vec<u8>, GpuTensor)> {
    let (m, _, _) = proj_dims(down);
    let y_bytes = iu4_y_bytes(gr, m, down);
    let t_tiles = up_i32(&mut cx.gpu, tiles_host)?;
    let t_y = up_bytes(&mut cx.gpu, &vec![CANARY_BYTE; y_bytes])?;
    let launched = k2_gemm(
        cx, codec, down, imp, table, &t_tiles, sorted, xq, &t_y, gr, src_rows,
    );
    let out = launched
        .and_then(|()| sync(&cx.gpu))
        .and_then(|()| down_bytes(&cx.gpu, &t_y));
    free_all(&mut cx.gpu, vec![t_tiles])?;
    match out {
        Ok(bytes) => Ok((bytes, t_y)),
        Err(err) => {
            free_all(&mut cx.gpu, vec![t_y])?;
            Err(err)
        }
    }
}

struct K2Out {
    gate: Vec<u8>,
    down: Vec<u8>,
}

/// gate/up K2 -> down A4 producer -> down K2 for one routing (host tiles/sorted), through the old
/// images (`old = Some`) or the new production launchers (`None`).
fn k2_pipeline(
    cx: &mut Ctx,
    codec: u8,
    old: Option<&[hip_bridge::Function]>,
    tabs: &Tables,
    rows: usize,
    x: &[f32],
    tiles: &[i32],
    sorted: &[i32],
) -> R<K2Out> {
    let slots = rows * K;
    let gr = grouped_rows_bound(slots);
    let t_sorted = up_i32(&mut cx.gpu, sorted)?;
    let t_x = up_f32(&mut cx.gpu, x)?;
    let xq_g = cx
        .gpu
        .qwen4_moe_rotate256_i4_slab(&t_x, GATE_K, rows)
        .map_err(e)?;
    let (gate, t_h) = k2_run(
        cx,
        codec,
        false,
        old.map(|f| &f[0]),
        &tabs.gate.table,
        tiles,
        &t_sorted,
        &xq_g,
        gr,
        rows,
    )?;
    let xq_d = cx
        .gpu
        .qwen4_moe_rotate128_i4_slab(&t_h, &t_sorted, DOWN_KK, gr, slots)
        .map_err(e)?;
    let (down, t_y) = k2_run(
        cx,
        codec,
        true,
        old.map(|f| &f[1]),
        &tabs.down.table,
        tiles,
        &t_sorted,
        &xq_d,
        gr,
        slots,
    )?;
    free_all(&mut cx.gpu, vec![t_sorted, t_x, t_h, t_y])?;
    Ok(K2Out { gate, down })
}

fn k2_phase(
    cx: &mut Ctx,
    rep: &mut Report,
    ph: &Phase,
    tabs: &Tables,
    fns: &[hip_bridge::Function],
    rng: &mut Rng,
) -> R<()> {
    for &rows in &K2_ROWS {
        let topk = make_topk(rng, rows, Spread::Skewed);
        let slots = rows * K;
        let lay = scatter_ref(&topk);
        let gr = lay.m_total_max;
        let t_sorted = up_i32(&mut cx.gpu, &lay.sorted)?;
        let x = rng.vec(rows * GATE_K, -2.0, 2.0);
        let t_x = up_f32(&mut cx.gpu, &x)?;
        let xq_g = cx
            .gpu
            .qwen4_moe_rotate256_i4_slab(&t_x, GATE_K, rows)
            .map_err(e)?;
        let tag = format!("{} K2 rows={rows}", ph.label);
        // gate/up: old image vs new launcher.
        let (old_g, t_old_g) = k2_run(
            cx,
            ph.codec,
            false,
            Some(&fns[0]),
            &tabs.gate.table,
            &lay.tiles,
            &t_sorted,
            &xq_g,
            gr,
            rows,
        )?;
        free_all(&mut cx.gpu, vec![t_old_g])?;
        let (new_g, t_h) = k2_run(
            cx,
            ph.codec,
            false,
            None,
            &tabs.gate.table,
            &lay.tiles,
            &t_sorted,
            &xq_g,
            gr,
            rows,
        )?;
        rep.cmp_bytes(&format!("{tag} gate_up new-vs-old bytes"), &new_g, &old_g);
        // down (input: the new gate/up output through the real A4 producer): old vs new.
        let xq_d = cx
            .gpu
            .qwen4_moe_rotate128_i4_slab(&t_h, &t_sorted, DOWN_KK, gr, slots)
            .map_err(e)?;
        let (old_d, t_old_d) = k2_run(
            cx,
            ph.codec,
            true,
            Some(&fns[1]),
            &tabs.down.table,
            &lay.tiles,
            &t_sorted,
            &xq_d,
            gr,
            slots,
        )?;
        free_all(&mut cx.gpu, vec![t_old_d])?;
        let (new_d, t_new_d) = k2_run(
            cx,
            ph.codec,
            true,
            None,
            &tabs.down.table,
            &lay.tiles,
            &t_sorted,
            &xq_d,
            gr,
            slots,
        )?;
        free_all(&mut cx.gpu, vec![t_new_d])?;
        rep.cmp_bytes(&format!("{tag} down new-vs-old bytes"), &new_d, &old_d);
        // Publication: the routed slots' real rows (every slot is routed in the valid run).
        let positions: Vec<usize> = lay.inverse.iter().map(|&p| p as usize).collect();
        if lay.inverse.iter().any(|&p| p < 0) {
            return Err(format!("{tag}: the valid layout left a slot unrouted"));
        }
        rep.publish_bf16_rows(
            &format!("{tag} gate_up"),
            &new_g,
            (GATE_M / 2) * 2,
            &positions,
        )?;
        rep.publish_bf16_rows(&format!("{tag} down"), &new_d, DOWN_MM * 2, &positions)?;

        if rows == K2_ROWS[0] {
            // Raw invalid tile ids (NEW launchers only): tiles of invalid experts stay canary.
            let live: Vec<usize> = (0..lay.tiles.len())
                .filter(|&t| lay.tiles[t] >= 0)
                .collect();
            if live.len() < 4 * BAD_IDS.len() {
                return Err(format!(
                    "{tag}: only {} live tiles, need {} for every bad id",
                    live.len(),
                    4 * BAD_IDS.len()
                ));
            }
            let mut tiles = lay.tiles.clone();
            let mut corrupted = vec![false; tiles.len()];
            for (j, &t) in live.iter().enumerate().filter(|(j, _)| j % 4 == 1) {
                tiles[t] = BAD_IDS[(j / 4) % BAD_IDS.len()];
                corrupted[t] = true;
            }
            for (down, xq, valid, t_valid) in
                [(false, &xq_g, &new_g, rows), (true, &xq_d, &new_d, slots)]
            {
                let (m, _, _) = proj_dims(down);
                let table = if down {
                    &tabs.down.table
                } else {
                    &tabs.gate.table
                };
                let tile_bytes = iu4_tile_bytes(gr, m, down);
                let label = format!(
                    "{tag} {} raw tile ids",
                    if down { "down" } else { "gate_up" }
                );
                match k2_run(
                    cx, ph.codec, down, None, table, &tiles, &t_sorted, xq, gr, t_valid,
                ) {
                    Ok((got, t_y)) => {
                        rep.bad_regions(
                            &label,
                            k2_tile_regions(&label, &got, valid, &corrupted, tile_bytes),
                            false,
                        );
                        free_all(&mut cx.gpu, vec![t_y])?;
                    }
                    Err(err) => rep.bad_launch_failed(&label, err),
                }
                // Teeth: a 256-entry view of the same valid physical table declares n_experts=256.
                let short = short_view(table, SHORT_N_EXPERTS);
                let skipped: Vec<bool> = lay
                    .tiles
                    .iter()
                    .map(|&id| id >= SHORT_N_EXPERTS as i32)
                    .collect();
                let label = format!(
                    "{tag} {} n_experts={SHORT_N_EXPERTS}",
                    if down { "down" } else { "gate_up" }
                );
                let (got, t_y) = k2_run(
                    cx, ph.codec, down, None, &short, &lay.tiles, &t_sorted, xq, gr, t_valid,
                )?;
                rep.teeth_regions(
                    &label,
                    k2_tile_regions(&label, &got, valid, &skipped, tile_bytes),
                );
                free_all(&mut cx.gpu, vec![t_y])?;
            }
        }
        free_all(&mut cx.gpu, vec![t_sorted, t_x])?;
    }
    Ok(())
}

impl Report {
    fn chain_regions(&mut self, label: &str, r: Result<(u64, u64), RegionFault>) {
        self.c.chain_runs += 1;
        match r {
            Ok((failed, valid)) => {
                self.c.chain_failed_slot_checks += failed;
                self.c.chain_valid_slot_compares += valid;
                self.results.push(Ok(format!(
                    "{label}: {failed} failed-row regions canary, {valid} valid regions == old"
                )));
            }
            Err(fault) => {
                self.c.chain_mismatches += 1;
                self.results.push(Err(fault.msg().to_owned()));
            }
        }
    }
}

/// Router (device) -> slab scatter (device) -> A4 producers -> K2 vs the old images over the
/// clean-twin routing. Every failed-row slot is unrouted (`inverse == -1`), every K2 tile the
/// scatter left unassigned stays canary, every valid slot's rows are bitwise the old chain's.
fn k2_chain(
    cx: &mut Ctx,
    rep: &mut Report,
    ph: &Phase,
    tabs: &Tables,
    fns: &[hip_bridge::Function],
    rng: &mut Rng,
) -> R<()> {
    let rows = CHAIN_ROWS;
    let slots = rows * K;
    let gr = grouped_rows_bound(slots);
    let x = rng.vec(rows * GATE_K, -2.0, 2.0);
    for (li, kinds) in chain_layouts(rows).iter().enumerate() {
        let (logits, clean, failing) = layout_logits(rng, kinds, li);
        let (old_ids, old_w) = run_router_old(cx, &clean, rows, false, true, true)?;
        let lay_old = scatter_ref(&old_ids);
        let old = k2_pipeline(
            cx,
            ph.codec,
            Some(fns),
            tabs,
            rows,
            &x,
            &lay_old.tiles,
            &lay_old.sorted,
        )?;
        for fast in [false, true] {
            if fast && !cx.fast_router_arch() {
                continue;
            }
            let tag = format!("{} K2 chain rows={rows} layout={li} fast={fast}", ph.label);
            let dev = router_device(cx, &logits, rows, fast, ROUTER_LAYER, ROUTER_ROW_BASE)?;
            let ids = down_i32(&cx.gpu, &dev.ids)?;
            let w = bits(&down_f32(&cx.gpu, &dev.w)?);
            let out = RouterOut {
                ids,
                w,
                fault: dev.fault,
            };
            router_check(
                rep,
                &tag,
                &out,
                &failing,
                &old_ids,
                &old_w,
                ROUTER_LAYER,
                ROUTER_ROW_BASE,
            );
            let b = scatter_bufs(&mut cx.gpu, slots, gr)?;
            cx.gpu
                .moe_scatter_stable_top10_slab(
                    &dev.ids, &b.counts, &b.offsets, &b.sorted, &b.tiles, &b.inverse, slots, E, gr,
                )
                .map_err(e)?;
            sync(&cx.gpu)?;
            let sc = b.download(&cx.gpu)?;
            b.free(&mut cx.gpu)?;
            free_all(&mut cx.gpu, vec![dev.ids, dev.w])?;
            let new = k2_pipeline(cx, ph.codec, None, tabs, rows, &x, &sc.tiles, &sc.sorted)?;
            rep.chain_regions(
                &format!("{tag} slots"),
                chain_slots(&tag, &new, &sc.inverse, &old, &lay_old.inverse, &failing),
            );
            // Tiles the scatter left unassigned must not be written by either K2 stage.
            let unassigned: Vec<bool> = sc.tiles.iter().map(|&t| t < 0).collect();
            for (down, bytes) in [(false, &new.gate), (true, &new.down)] {
                let (m, _, _) = proj_dims(down);
                let label = format!(
                    "{tag} {} unassigned tiles",
                    if down { "down" } else { "gate_up" }
                );
                rep.chain_regions(
                    &label,
                    k2_tile_regions(
                        &label,
                        bytes,
                        bytes,
                        &unassigned,
                        iu4_tile_bytes(gr, m, down),
                    ),
                );
            }
        }
    }
    Ok(())
}

/// Per slot of the new chain: a failed row's slots are unrouted; every other slot's gate/up and
/// down rows equal the old chain's rows of the same slot (positions differ between the layouts).
fn chain_slots(
    label: &str,
    new: &K2Out,
    inv_new: &[i32],
    old: &K2Out,
    inv_old: &[i32],
    failing: &[bool],
) -> Result<(u64, u64), RegionFault> {
    let (gate_row, down_row) = ((GATE_M / 2) * 2, DOWN_MM * 2);
    let (mut failed, mut valid) = (0u64, 0u64);
    for slot in 0..inv_new.len() {
        if failing[slot / K] {
            if inv_new[slot] != -1 {
                return Err(RegionFault::Stored(format!(
                    "{label}: failed-row slot {slot} was routed (inverse {})",
                    inv_new[slot]
                )));
            }
            failed += 1;
            continue;
        }
        let (pn, po) = (inv_new[slot], inv_old[slot]);
        if pn < 0 || po < 0 {
            return Err(RegionFault::Valid(format!(
                "{label}: valid slot {slot} unrouted (new {pn}, old {po})"
            )));
        }
        let (pn, po) = (pn as usize, po as usize);
        for (nb, ob, rb, what) in [
            (&new.gate, &old.gate, gate_row, "gate_up"),
            (&new.down, &old.down, down_row, "down"),
        ] {
            if nb[pn * rb..(pn + 1) * rb] != ob[po * rb..(po + 1) * rb] {
                return Err(RegionFault::Valid(format!(
                    "{label}: valid slot {slot} {what} row differs from the old chain"
                )));
            }
        }
        valid += 1;
    }
    Ok((failed, valid))
}

// ───────────────────────────────────────────────────────────────────────────────
// Router producer
// ───────────────────────────────────────────────────────────────────────────────
struct RouterDev {
    ids: GpuTensor,
    w: GpuTensor,
    fault: Option<RouterFault>,
}

/// The production router launch whose ids/weights stay on the device (the consumer chain's input).
fn router_device(
    cx: &mut Ctx,
    logits: &[f32],
    rows: usize,
    fast: bool,
    layer: usize,
    row_base: usize,
) -> R<RouterDev> {
    Arc::make_mut(&mut cx.gpu.flags).qwen4_router_fast = fast;
    cx.gpu.clear_router_fault();
    let t_l = up_f32(&mut cx.gpu, logits)?;
    let ids = int_fill(&mut cx.gpu, rows * K, -2)?;
    let w = sentinel_f32(&mut cx.gpu, rows * K)?;
    cx.gpu
        .moe_router_softmax_top10_f32(&t_l, &ids, &w, rows, true, true, layer, row_base)
        .map_err(e)?;
    sync(&cx.gpu)?;
    let fault = cx.gpu.peek_router_fault();
    cx.gpu.clear_router_fault();
    free_all(&mut cx.gpu, vec![t_l])?;
    Ok(RouterDev { ids, w, fault })
}

/// Router row layouts of a chain of `rows` rows: every third row bad, all rows bad but one, every
/// row bad, no row bad (kinds cycle through the all-NaN / +Inf / -Inf router rows).
fn chain_layouts(rows: usize) -> Vec<Vec<Option<Bad>>> {
    let kind = |r: usize| ROUTER_BADS[r % ROUTER_BADS.len()];
    vec![
        (0..rows).map(|r| (r % 3 == 1).then(|| kind(r))).collect(),
        (0..rows)
            .map(|r| (r != rows / 2).then(|| kind(r)))
            .collect(),
        (0..rows).map(|r| Some(kind(r))).collect(),
        vec![None; rows],
    ]
}

/// `(logits, clean twin, failing mask)` of a layout: the clean twin replaces failing rows with
/// finite ones, everything else identical.
fn layout_logits(
    rng: &mut Rng,
    kinds: &[Option<Bad>],
    salt: usize,
) -> (Vec<f32>, Vec<f32>, Vec<bool>) {
    let mut logits = Vec::with_capacity(kinds.len() * E);
    for (r, kind) in kinds.iter().enumerate() {
        logits.extend(match kind {
            Some(k) => bad_row(rng, *k),
            None => finite_row(rng, r + salt),
        });
    }
    let failing: Vec<bool> = (0..kinds.len())
        .map(|r| row_fails(&logits[r * E..(r + 1) * E], true))
        .collect();
    let mut clean = logits.clone();
    for r in 0..kinds.len() {
        if failing[r] {
            clean[r * E..(r + 1) * E].copy_from_slice(&finite_row(rng, r + 3));
        }
    }
    (logits, clean, failing)
}

/// The fail-closed contract of one router launch (ids -1 / weights +0 on failing rows, finite rows
/// bitwise the pinned beta router's) and its typed fault word: layer exact, global row exact for a
/// single failing row, one failing row when several race in one launch.
fn router_check(
    rep: &mut Report,
    label: &str,
    out: &RouterOut,
    failing: &[bool],
    old_ids: &[i32],
    old_w: &[u32],
    layer: usize,
    row_base: usize,
) {
    let n_bad = failing.iter().filter(|f| **f).count();
    rep.c.router_launches += 1;
    rep.c.router_failed_rows += n_bad as u64;
    rep.c.router_finite_rows += (failing.len() - n_bad) as u64;
    let rows = check_router_rows(label, out, failing, old_ids, old_w);
    if rows.is_err() {
        rep.c.router_contract_mismatches += 1;
    }
    rep.results.push(rows);
    let first = failing.iter().position(|f| *f);
    let fault = match (first, out.fault) {
        (None, None) => Ok(format!("{label}: no failing row, no fault")),
        (None, Some(f)) => Err(format!("{label}: no failing row but fault {f:?}")),
        (Some(_), None) => Err(format!("{label}: failing rows but the fault word is clear")),
        (Some(first), Some(f)) => {
            let row_ok = if n_bad == 1 {
                f.row == (row_base + first) as u64
            } else {
                failing
                    .get((f.row as usize).wrapping_sub(row_base))
                    .copied()
                    .unwrap_or(false)
            };
            if f.layer == layer as u32 && row_ok {
                Ok(format!("{label}: fault layer={} row={}", f.layer, f.row))
            } else {
                Err(format!(
                    "{label}: fault {f:?}, want layer {layer} and a failing row (first bad {})",
                    row_base + first
                ))
            }
        }
    };
    if fault.is_err() {
        rep.c.router_fault_mismatches += 1;
    }
    rep.results.push(fault);
}

/// Single failing row of every router kind at several rows, standard and fast kernel: the fault
/// word is EXACTLY `(layer, row_base + row)`, the row fails closed, every other row is bitwise
/// the pinned beta router's. Then the teeth: the pinned beta router must NOT fail an all-NaN row closed.
fn router_exact(cx: &mut Ctx, rep: &mut Report, rng: &mut Rng) -> R<()> {
    let rows = 5usize;
    for &kind in &ROUTER_BADS {
        for &bad_row_ix in &[0usize, 2, 4] {
            let kinds: Vec<Option<Bad>> = (0..rows)
                .map(|r| (r == bad_row_ix).then_some(kind))
                .collect();
            let (logits, clean, failing) = layout_logits(rng, &kinds, bad_row_ix);
            let (old_ids, old_w) = run_router_old(cx, &clean, rows, false, true, true)?;
            for fast in [false, true] {
                if fast && !cx.fast_router_arch() {
                    continue;
                }
                let out = run_router_new(
                    cx,
                    &logits,
                    rows,
                    fast,
                    true,
                    true,
                    ROUTER_LAYER,
                    ROUTER_ROW_BASE,
                )?;
                router_check(
                    rep,
                    &format!("router {kind:?} row={bad_row_ix} fast={fast}"),
                    &out,
                    &failing,
                    &old_ids,
                    &old_w,
                    ROUTER_LAYER,
                    ROUTER_ROW_BASE,
                );
            }
        }
    }
    let kinds = vec![None, Some(Bad::AllNan), None];
    let (logits, _, _) = layout_logits(rng, &kinds, 0);
    let (ids, w) = run_router_old(cx, &logits, 3, false, true, false)?;
    let met = ids[K..2 * K].iter().all(|&v| v == -1) && w[K..2 * K].iter().all(|&v| v == 0);
    if met {
        rep.results.push(Err("the pinned beta router already fails an all-NaN row closed: the router assertions have no teeth".into()));
    } else {
        rep.c.router_old_teeth_controls += 1;
        rep.results.push(Ok(
            "pinned beta router on an all-NaN row does not meet the fail-closed contract (teeth)"
                .into(),
        ));
    }
    Ok(())
}

// ───────────────────────────────────────────────────────────────────────────────
// Resource model
// ───────────────────────────────────────────────────────────────────────────────
/// Transient device bytes of one K2 stage set at `rows` prefill rows (x, tiles, sorted, the two
/// A4 sidecars, the gate/up and down BF16 outputs alive together).
fn k2_transient(rows: usize) -> u64 {
    let slots = rows * K;
    let gr = grouped_rows_bound(slots);
    let (x, tiles, sorted) = (rows * GATE_K * 4, gr / BLOCK_M * 4, gr * 4);
    let sidecars = (GATE_K / 128) * rows * 72 + (DOWN_KK / 128) * slots * 72;
    let y = gr * (GATE_M / 2) * 2 + gr * DOWN_MM * 2;
    (x + tiles + sorted + sidecars + y) as u64
}

/// Transient device bytes of one K1 launch at `lanes` lanes (worst projection: x, y0, y1, union
/// copies, topk).
fn k1_transient(lanes: usize) -> u64 {
    let s = K * lanes;
    let gate = lanes * GATE_K * 4 + 2 * s * (GATE_M / 2) * 4;
    let down = s * DOWN_KK * 4 + s * DOWN_MM * 4;
    let union = 2 * (6 * s + 2) * 4;
    (gate.max(down) + union + s * 4) as u64
}

/// Extra device bytes of the router -> scatter chain at `rows` rows (logits, ids, weights, scatter buffers).
fn chain_extra(rows: usize) -> u64 {
    let slots = rows * K;
    let gr = grouped_rows_bound(slots);
    (rows * E * 4 + 2 * slots * 4 + E * 4 + (E + 1) * 4 + gr * 4 + gr / BLOCK_M * 4 + slots * 4)
        as u64
}

fn phase_weights(codec: u8) -> u64 {
    let n = N_REAL as u64;
    let (gate, down) = match codec {
        SLAB_CODEC_E8P => (
            e8p_blob_bytes(GATE_M, GATE_K),
            e8p_blob_bytes(DOWN_MM, DOWN_KK),
        ),
        _ => (
            mq4_gate_bytes(GATE_M, GATE_K),
            mq4_down_bytes(DOWN_MM, DOWN_KK),
        ),
    };
    n * (gate + down) as u64 + 2 * (E * 8) as u64
}

fn peak_transient() -> u64 {
    let k2 = K2_ROWS.iter().map(|&r| k2_transient(r)).max().unwrap_or(0);
    let chain = k2_transient(CHAIN_ROWS) + chain_extra(CHAIN_ROWS);
    k1_transient(*LANES.last().unwrap_or(&1)).max(k2).max(chain)
}

/// Peak host bytes: the old+new K2 byte vectors of the widest rows (gate/up and down), the
/// published f32 vector, its little-endian file image, and the widest expert blob under generation.
fn host_peak(out_elements: usize) -> u64 {
    let rows = *K2_ROWS.last().unwrap_or(&1);
    let gr = grouped_rows_bound(rows * K);
    let k2_bytes = 2 * (gr * (GATE_M / 2) * 2 + gr * DOWN_MM * 2);
    let blob = 2 * mq4_gate_bytes(GATE_M, GATE_K).max(e8p_blob_bytes(GATE_M, GATE_K));
    (k2_bytes + blob + 2 * out_elements * 4) as u64
}

fn resource_model(arch: &str, out_elements: usize) -> Value {
    let phases: Vec<(&str, u8)> = if arch == "gfx1100" {
        vec![("E8P", SLAB_CODEC_E8P), ("MQ4-hot", SLAB_CODEC_MQ4)]
    } else {
        vec![("E8P", SLAB_CODEC_E8P)]
    };
    let union_scratch = rdna_compute::scratch::qwen4_slab_descriptor_bytes(slab_moe::SLAB_MAX_LANES)
        .expect("the fixed slab lane capacity has a valid descriptor extent")
        as u64;
    let transient = peak_transient();
    let per_phase: Vec<Value> = phases
        .iter()
        .map(|&(name, codec)| {
            let weights = phase_weights(codec);
            json!({"phase": name, "weights_and_tables_bytes": weights, "peak_bytes": weights + transient + union_scratch})
        })
        .collect();
    let device_peak = per_phase
        .iter()
        .filter_map(|p| p["peak_bytes"].as_u64())
        .max()
        .unwrap_or(0);
    let host = host_peak(out_elements);
    json!({
        "loads_flash_next": false,
        "model": "analytic live-set model of this case's own allocations (phases free their tables before the next phase); excludes the HIP context, loaded code objects and Gpu::init scratch",
        "real_experts_per_projection": N_REAL,
        "pointer_table_entries": E,
        "k2_rows": K2_ROWS,
        "k1_lanes": LANES,
        "chain_rows": CHAIN_ROWS,
        "k1_transient_bytes_8_lanes": k1_transient(8),
        "k2_transient_bytes": K2_ROWS.iter().map(|&r| json!({"rows": r, "bytes": k2_transient(r)})).collect::<Vec<_>>(),
        "chain_transient_bytes": k2_transient(CHAIN_ROWS) + chain_extra(CHAIN_ROWS),
        "slab_union_scratch_bytes": union_scratch,
        "phases": per_phase,
        "peak_device_bytes": device_peak,
        "peak_host_bytes": host,
    })
}

// ───────────────────────────────────────────────────────────────────────────────
// Entry point
// ───────────────────────────────────────────────────────────────────────────────
fn out_paths(cx: &Ctx) -> (PathBuf, PathBuf) {
    (
        cx.out_dir.join(format!("{CASE_NAME}.{}.json", cx.arch)),
        cx.out_dir.join(format!("{CASE_NAME}.{}.f32le", cx.arch)),
    )
}

pub(super) fn run(cx: &mut Ctx) -> R<Outcome> {
    if !rdna_compute::qwen4_slab_codec_admitted(&cx.arch, SLAB_CODEC_E8P) {
        return Ok(Outcome::Skip(format!(
            "{CASE_NAME}: the slab matrix serves E8P on gfx1201, gfx1100 and gfx1151 only, device is {}",
            cx.arch
        )));
    }
    std::fs::create_dir_all(&cx.out_dir)
        .map_err(|err| format!("create {}: {err}", cx.out_dir.display()))?;
    let (json_path, f32_path) = out_paths(cx);
    for p in [&json_path, &f32_path] {
        match std::fs::remove_file(p) {
            Ok(()) => {}
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => return Err(format!("remove stale {}: {err}", p.display())),
        }
    }
    if cx.old_dir.is_none() {
        return Err(format!(
            "{CASE_NAME}: needs --old-dir: the rc4b E8P/MQ4 images are not tracked at the pinned beta ref (gfx1151: the pre-rc5 images of commit 11411610b)"
        ));
    }
    let mut rep = Report::default();
    let body = run_body(cx, &mut rep);
    finish(cx, rep, body)
}

fn run_body(cx: &mut Ctx, rep: &mut Report) -> R<()> {
    let mut rng = Rng::new(0x4E41_4EE8);
    cx.gpu
        .ensure_qwen4_slab_kernels(SLAB_CODEC_E8P)
        .map_err(e)?;
    cx.gpu.reserve_qwen4_slab(0).map_err(e)?;
    rep.source(cx, &OLD_ROUTER)?;
    router_exact(cx, rep, &mut rng)?;
    let mut codecs = vec![SLAB_CODEC_E8P];
    if cx.arch == "gfx1100" {
        codecs.push(SLAB_CODEC_MQ4);
    }
    for codec in codecs {
        let ph = phase(cx, codec)?;
        rep.phases.push(ph.label.to_owned());
        let tabs = Tables::build(cx, &mut rng, codec)?;
        k1_phase(cx, rep, &ph, &tabs, &mut rng)?;
        let (_k2_module, fns) = cx.load_old_image(&ph.k2_old_rel, &ph.k2_syms)?;
        rep.image(cx, &ph.k2_old_rel)?;
        k2_phase(cx, rep, &ph, &tabs, &fns, &mut rng)?;
        if codec == SLAB_CODEC_E8P {
            k1_chain(cx, rep, &ph, &tabs, &mut rng)?;
            k2_chain(cx, rep, &ph, &tabs, &fns, &mut rng)?;
        }
        tabs.free(cx)?;
    }
    Ok(())
}

fn finish(cx: &mut Ctx, rep: Report, body: R<()>) -> R<Outcome> {
    let (json_path, f32_path) = out_paths(cx);
    let Report {
        c,
        results,
        out,
        segments,
        images,
        sources,
        phases,
    } = rep;
    let verdict_err = body.err().or_else(|| all(results).err()).or_else(|| {
        (!c.accepted()).then(|| format!("acceptance counters not met: {}", c.to_json()))
    });
    let pass = verdict_err.is_none();
    let mut file_bytes = Vec::new();
    if pass {
        file_bytes.reserve(out.len() * 4);
        for v in &out {
            file_bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(&f32_path, &file_bytes)
            .map_err(|err| format!("write {}: {err}", f32_path.display()))?;
    }
    let report = json!({
        "schema": SCHEMA,
        "case": CASE_NAME,
        "arch": cx.arch,
        "verdict": if pass { "PASS" } else { "FAIL" },
        "error": verdict_err,
        "skipped": false,
        "argv": std::env::args().collect::<Vec<_>>(),
        "old_ref": cx.old_ref,
        "old_dir": cx.old_dir.as_ref().map(|d| d.display().to_string()),
        "phases": phases,
        "old_images": images,
        "old_sources": sources,
        "old_sources_note": "every old source this case compiles: only moe_router_softmax_top10_f32.hip (run_router_old always passes fast=false, also against the fast router's new output on gfx1201 and gfx1151); moe_router_softmax_top10_f32_fast.hip and the old planner are never loaded",
        "counters": c.to_json(),
        "output": {
            "file": f32_path.file_name().map(|n| n.to_string_lossy().into_owned()),
            "published": pass,
            "dtype": "f32le",
            "elements": c.output_elements,
            "bytes": c.output_elements * 4,
            "nonfinite": c.output_nonfinite,
            "sha256": if pass { Some(sha256_hex(&file_bytes)) } else { None },
            "segments": segments.iter().map(|s| json!({
                "label": s.label, "source": s.source, "offset": s.offset, "elements": s.elements, "sha256": s.sha256,
            })).collect::<Vec<_>>(),
        },
        "resources": resource_model(&cx.arch, out.len()),
    });
    let text = serde_json::to_string_pretty(&report).map_err(e)?;
    std::fs::write(&json_path, text + "\n")
        .map_err(|err| format!("write {}: {err}", json_path.display()))?;
    cx.notes.push(format!(
        "report {} output {}",
        json_path.display(),
        f32_path.display()
    ));
    match verdict_err {
        None => Ok(Outcome::Pass(format!(
            "{} finite exact compares ({} B), {} raw bad-id cases, {} positive + {} teeth controls, router containment {}, {} output f32 elements all finite -> {}",
            c.finite_exact_compares,
            c.finite_exact_compare_bytes,
            c.raw_bad_id_cases,
            c.positive_controls,
            c.teeth_controls,
            c.router_containment(),
            c.output_elements,
            f32_path.display()
        ))),
        Some(err) => Err(err),
    }
}

// ───────────────────────────────────────────────────────────────────────────────
// CPU tests (no GPU)
// ───────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn k1(y0: Vec<u32>, y1: Vec<u32>) -> K1Out {
        K1Out { y0, y1 }
    }

    #[test]
    fn blob_geometry_matches_the_planar_wire() {
        assert_eq!(e8p_blob_bytes(GATE_M, GATE_K), 870_400);
        assert_eq!(e8p_blob_bytes(DOWN_MM, DOWN_KK), 435_200);
        assert_eq!(mq4_gate_bytes(GATE_M, GATE_K), 1_740_800);
        assert_eq!(mq4_down_bytes(DOWN_MM, DOWN_KK), 870_400);
        let mut rng = Rng::new(1);
        for mode in 0..2 {
            let blob = e8p_blob(&mut rng, 64, 256, mode);
            assert_eq!(blob.len(), e8p_blob_bytes(64, 256));
            let words_end = 64 * 256 / 4;
            for d in blob[words_end..].chunks_exact(2) {
                assert!(e8p::writer_guard(u16::from_le_bytes([d[0], d[1]])).is_ok());
            }
        }
    }

    /// Every arch whose slab matrix serves E8P has an old K2 baseline stem named after its own
    /// arch (the K1 old image is named after the arch's cold K1 module), and only gfx1100 has the
    /// hot-MQ4 old K2; no other arch or codec invents one.
    #[test]
    fn old_baseline_images_are_arch_keyed_and_cover_every_e8p_arch() {
        for arch in ["gfx1201", "gfx1100", "gfx1151"] {
            let stem = k2_old_stem(arch, SLAB_CODEC_E8P).unwrap();
            assert_eq!(stem, format!("qwen4_moe_iu4_sym_e8p_pm_{arch}"));
            let k1 = slab_moe::slab_cold_k1_metadata(arch, SLAB_CODEC_E8P).unwrap();
            assert!(k1.k1.module.ends_with(arch), "{arch}: {}", k1.k1.module);
        }
        assert_eq!(
            k2_old_stem("gfx1100", SLAB_CODEC_MQ4),
            Ok("qwen4_moe_iu4_sym_pm_gfx1100")
        );
        for arch in ["gfx1201", "gfx1151"] {
            assert!(k2_old_stem(arch, SLAB_CODEC_MQ4).is_err(), "{arch}");
        }
        for arch in ["gfx942", "gfx1200", "gfx1101", "gfx1150", ""] {
            assert!(k2_old_stem(arch, SLAB_CODEC_E8P).is_err(), "{arch:?}");
        }
        // Resource model: only gfx1100 adds the hot-MQ4 phase.
        for (arch, phases) in [("gfx1201", 1), ("gfx1151", 1), ("gfx1100", 2)] {
            assert_eq!(
                resource_model(arch, 1)["phases"].as_array().unwrap().len(),
                phases,
                "{arch}"
            );
        }
    }

    #[test]
    fn canary_is_finite_and_not_the_nan_poison() {
        assert!(f32::from_bits(CANARY_WORD).is_finite());
        assert_ne!(CANARY_WORD, nan_poison().to_bits());
        assert!(bf16_bits_to_f32(u16::from_le_bytes([CANARY_BYTE; 2])).is_finite());
    }

    #[test]
    fn k1_regions_split_canary_and_valid() {
        let ys = 3;
        let want = k1(vec![1, 2, 3, 4, 5, 6], vec![]);
        let ok = k1(vec![1, 2, 3, CANARY_WORD, CANARY_WORD, CANARY_WORD], vec![]);
        assert_eq!(
            k1_regions("t", &ok, &want, &[false, true], ys, CANARY_WORD).ok(),
            Some((1, 1))
        );
        let stored = k1(vec![1, 2, 3, CANARY_WORD, 9, CANARY_WORD], vec![]);
        assert!(matches!(
            k1_regions("t", &stored, &want, &[false, true], ys, CANARY_WORD),
            Err(RegionFault::Stored(_))
        ));
        let changed = k1(vec![1, 2, 7, CANARY_WORD, CANARY_WORD, CANARY_WORD], vec![]);
        assert!(matches!(
            k1_regions("t", &changed, &want, &[false, true], ys, CANARY_WORD),
            Err(RegionFault::Valid(_))
        ));
        // A zero-filled skipped region is NOT canary: the native PM skip never zero-fills.
        let zero = k1(vec![1, 2, 3, 0, 0, 0], vec![]);
        assert!(matches!(
            k1_regions("t", &zero, &want, &[false, true], ys, CANARY_WORD),
            Err(RegionFault::Stored(_))
        ));
    }

    #[test]
    fn k2_tile_regions_split_canary_and_valid() {
        let want = vec![1u8, 2, 3, 4];
        let ok = vec![1u8, 2, CANARY_BYTE, CANARY_BYTE];
        assert_eq!(
            k2_tile_regions("t", &ok, &want, &[false, true], 2).ok(),
            Some((1, 1))
        );
        let zero = vec![1u8, 2, 0, 0];
        assert!(matches!(
            k2_tile_regions("t", &zero, &want, &[false, true], 2),
            Err(RegionFault::Stored(_))
        ));
        let valid = vec![9u8, 2, CANARY_BYTE, CANARY_BYTE];
        assert!(matches!(
            k2_tile_regions("t", &valid, &want, &[false, true], 2),
            Err(RegionFault::Valid(_))
        ));
    }

    #[test]
    fn chain_layouts_cover_every_router_kind_and_mixed_rows() {
        let layouts = chain_layouts(CHAIN_ROWS);
        assert_eq!(layouts.len(), 4);
        assert!(layouts.iter().all(|l| l.len() == CHAIN_ROWS));
        let mut rng = Rng::new(3);
        for (li, kinds) in layouts.iter().enumerate() {
            let (logits, clean, failing) = layout_logits(&mut rng, kinds, li);
            assert_eq!(logits.len(), CHAIN_ROWS * E);
            for r in 0..CHAIN_ROWS {
                assert_eq!(failing[r], kinds[r].is_some(), "layout {li} row {r}");
                assert!(!row_fails(&clean[r * E..(r + 1) * E], true));
            }
        }
        assert!(layouts[0].iter().any(|k| k.is_some()) && layouts[0].iter().any(|k| k.is_none()));
        assert!(layouts[2].iter().all(|k| k.is_some()) && layouts[3].iter().all(|k| k.is_none()));
        // Every router kind of the contract appears in the all-bad layout cycle.
        let kinds: Vec<String> = layouts[2]
            .iter()
            .flatten()
            .map(|k| format!("{k:?}"))
            .collect();
        for want in ["AllNan", "PlusInf", "MinusInf"] {
            assert!(kinds.iter().any(|k| k == want), "{want}");
        }
    }

    #[test]
    fn short_declared_count_cuts_inside_the_expert_range() {
        assert!(SHORT_N_EXPERTS < E && SHORT_N_EXPERTS > 0);
        // Resource model terms are closed forms of the fixtures.
        assert_eq!(
            phase_weights(SLAB_CODEC_E8P),
            16 * (870_400 + 435_200) + 2 * 4096
        );
        assert_eq!(
            phase_weights(SLAB_CODEC_MQ4),
            16 * (1_740_800 + 870_400) + 2 * 4096
        );
        assert_eq!(grouped_rows_bound(330), 5280);
        assert_eq!(
            k2_transient(33),
            (33 * 2560 * 4
                + 330 * 4
                + 5280 * 4
                + 20 * 33 * 72
                + 5 * 330 * 72
                + 5280 * 640 * 2
                + 5280 * 2560 * 2) as u64
        );
    }

    #[test]
    fn publication_names_and_schema_are_fixed() {
        assert_eq!(SCHEMA, "c24-e8p-nan-producer-contract-v1");
        assert_eq!(CASE_NAME, "e8p_nan_containment");
        let mut rep = Report::default();
        rep.publish_f32("t", "f32", [1.0f32, f32::NAN, 2.0].into_iter());
        assert_eq!((rep.c.output_elements, rep.c.output_nonfinite), (3, 1));
        assert!(!rep.c.accepted());
    }

    /// The planner poison phases over every K1 lane count: a fully failed-closed one-lane row
    /// (LANES[0], phase 1) has all slots invalid and no valid region, which is legitimate; every
    /// other (lanes, phase) keeps both canary and valid regions. Strictness is preserved: the
    /// all-failed allowance never accepts a partial mask, and a fully failed mask is not accepted
    /// as an ordinary (non-all-failed) case.
    #[test]
    fn planner_phases_accept_all_failed_rows_only_when_every_slot_is_invalid() {
        let ys = 3;
        let mut rng = Rng::new(5);
        let (mut all_failed_masks, mut partial_masks) = (0u64, 0u64);
        for &lanes in &LANES {
            let topk = straddling_topk(&mut rng, lanes).unwrap();
            let slots = K * lanes;
            let want = k1(
                (0..(slots * ys) as u32).map(|v| v + 1).collect(),
                (0..(slots * ys) as u32).map(|v| v + 1000).collect(),
            );
            let mut lane_partial = 0u64;
            for phase_id in 0..5 {
                let (_, bad) = poison_topk(&topk, phase_id);
                let all_failed = bad.iter().all(|&b| b);
                let canary_of = |w: &[u32]| -> Vec<u32> {
                    w.iter()
                        .enumerate()
                        .map(|(i, &v)| if bad[i / ys] { CANARY_WORD } else { v })
                        .collect()
                };
                let got = k1(canary_of(&want.y0), canary_of(&want.y1));
                let region = || k1_regions("t", &got, &want, &bad, ys, CANARY_WORD);
                let mut rep = Report::default();
                rep.bad_regions("t", region(), all_failed);
                assert!(
                    rep.results.iter().all(|r| r.is_ok()),
                    "lanes={lanes} phase={phase_id} all_failed={all_failed}: {:?}",
                    rep.results
                );
                assert_eq!(rep.c.all_failed_cases, u64::from(all_failed));
                assert_eq!(rep.c.raw_bad_id_valid_regions == 0, all_failed);
                assert!(rep.c.raw_bad_id_canary_regions > 0);
                // Strictness: the wrong all_failed claim is rejected.
                let mut wrong = Report::default();
                wrong.bad_regions("t", region(), !all_failed);
                assert!(
                    wrong.results.iter().any(|r| r.is_err()),
                    "lanes={lanes} phase={phase_id}"
                );
                if all_failed {
                    all_failed_masks += 1;
                } else {
                    partial_masks += 1;
                    lane_partial += 1;
                }
            }
            assert!(
                lane_partial > 0,
                "lanes={lanes}: no phase keeps a valid region"
            );
        }
        assert!(
            all_failed_masks > 0,
            "the one-lane all-failed phase is not covered"
        );
        assert!(partial_masks > 0);
    }
}
