// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Frozen one-row decode step of Inkling-2-Mini (arch 24): stage, body, finish.
//!
//! The incumbent one-row decode ([`crate::forward_decode_gpu`], the blocked
//! body at one row) allocates its visit temporaries, uploads host gather maps
//! and bakes the host position into its launches, which a retained tape
//! cannot replay. This module runs the **same** 48-visit body and head
//! (`gpu_prefill::frozen_visit` / `frozen_head` share the arithmetic and the
//! launch order with the incumbent) over a dedicated per-load M=1
//! [`DecodeArena`]: stable addresses, no allocation, no host position, no
//! host tag or cursor mutation inside the body.
//!
//! ```text
//! plan_decode_step     host validation, bucket + key          (no GPU work)
//! stage_decode_input   ONE H2D: [pos:u32, pad16, packed qt44 embedding row]
//! enqueue_decode_body  embed unpack + norm, 48 visits, head + softcap
//!                      (enqueue only: no alloc/free, H2D/D2H, sync, host commit)
//! finish_decode_step   ONE D2H of the unpadded logits, THEN commit tags,
//!                      cursors and `seq_pos` (an enqueue is not a commit)
//! ```
//!
//! [`forward_decode_frozen`] composes the four eagerly; it is the frozen-eager
//! arm the oracle compares bitwise against stock eager (`forward_decode_gpu`).

use crate::arch_model::Inkling2Bundle;
use crate::config::Inkling2Config;
use crate::gpu::{Inkling2GpuState, Inkling2GpuWeights, INKLING2_KV_DTYPE};
use crate::gpu_forward::ensure_kernels;
use crate::gpu_prefill::{
    check_pre_block_tags, ensure_pf_kernels, frozen_copy, frozen_head, frozen_scope_enter,
    frozen_scope_exit, frozen_visit, prefill_supported, FrozenScratch, FrozenVisit, Msgs,
    PrefillScratch,
};
use hipfire_dispatch::pipeline::PrefillActTier;
use hipfire_runtime::llama::weight_gemv;
use rdna_compute::inkling2_attention::{inkling2_attn_fa2_splits, INKLING2_ATTN_MAX_SPLITS};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::cell::RefCell;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Visits of a Mini decode step (the fixed per-visit arrays of a bucket).
pub const MAX_VISITS: usize = 48;

/// Revision of the position-ABI / split policy / bank layout this body bakes;
/// bump on any change a recorded tape would not survive.
pub const DECODE_ABI_REV: u32 = 1;

/// Key-count cuts of the buckets (inclusive upper bounds); the position limit
/// is appended as the final bound and every cut is clipped to it.
const BUCKET_CUTS: [usize; 4] = [128, 192, 2048, 4096];

/// Bytes of the leading position word of the staged input record.
const REC_HEAD: usize = 16;

/// Arena region alignment (the bump arena's carve alignment).
const ALIGN: usize = 256;

// ---------------------------------------------------------------------------
// Static schedule
// ---------------------------------------------------------------------------

/// A `[n_embd]` F32 row of the arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Slot {
    /// Residual stream ping-pong.
    X(u8),
    /// Stream after the prelude (the base of every pass).
    PreludeX,
    /// Stream at the end of the previous pass.
    PrevOut,
    /// Prelude block bank.
    Pre(u8),
    /// Previous-pass block bank.
    Prev(u8),
    /// Current block bank (what the visit's AttnRes mix reads).
    Cur(u8),
}

/// One bank copy (the stock `dup`, into its fixed destination).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CopyOp {
    pub src: Slot,
    pub dst: Slot,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Phase {
    Prelude,
    Pass(u8),
    Post,
}

/// One of the 48 visits, fully static for a load.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct VisitDesc {
    pub visit: u8,
    /// Physical layer whose weights the visit runs.
    pub phys: u8,
    pub swa: bool,
    /// K/V ring rows of this visit's own cache (constant for the load).
    pub ring_rows: u32,
    pub phase: Phase,
    /// Controller / cross-pass edge index (pass > 0 only).
    pub core_i: Option<u8>,
    pub x_in: u8,
    pub x_out: u8,
    /// Live entries of the current block bank at visit start.
    pub blocks_len: u8,
    /// `(skip, len)` of the previous-pass bank the pass messages mix over.
    pub prev: Option<(u8, u8)>,
    /// Current-bank entry receiving this visit's block push.
    pub push: Option<u8>,
    /// Bank copies run before the visit (prelude snapshot, pass boundaries).
    pub before: Vec<CopyOp>,
}

/// The typed static 48-visit schedule of one load.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DecodeSchedule {
    pub visits: Vec<VisitDesc>,
    pub n_pre: u8,
    pub n_prev: u8,
    pub n_cur: u8,
    /// x slot holding the stream after the last visit.
    pub x_final: u8,
}

impl DecodeSchedule {
    /// Build the schedule from the loop geometry and the load's per-visit rings.
    /// Mirrors `gpu_prefill::drive_block`/`step` op for op.
    pub fn build(cfg: &Inkling2Config, visit_rows: &[usize]) -> Result<Self, String> {
        let lc = &cfg.loop_cfg;
        let n_slot = lc.n_slot();
        if n_slot == 0 || n_slot > MAX_VISITS || visit_rows.len() != n_slot {
            return Err(format!(
                "inkling2 decode: {n_slot} visits / {} rings (max {MAX_VISITS})",
                visit_rows.len()
            ));
        }
        let block = cfg.attn_res_block;
        if block == 0 {
            return Err("inkling2 decode: attn_res_block is 0".into());
        }
        let core_len = lc.core_end - lc.core_start;
        let looped = lc.num_passes * core_len;
        let pushes = |r: std::ops::Range<usize>| r.filter(|p| p % block == 0).count();
        let np = pushes(0..lc.core_start);
        let nc = pushes(lc.core_start..lc.core_end);
        let npost = pushes(lc.core_end..lc.n_phys);
        let (n_pre, n_prev, n_cur) = (np, np + nc, np + nc + npost);
        if n_cur > u8::MAX as usize {
            return Err("inkling2 decode: block bank too large".into());
        }
        let mut visits = Vec::with_capacity(n_slot);
        let mut cur_len = 0usize;
        let mut x = 0u8;
        for v in 0..n_slot {
            let phys = lc.layer_phys(v);
            let in_core = v >= lc.core_start && v < lc.core_start + looped;
            let pass = if in_core {
                Some((v - lc.core_start) / core_len)
            } else {
                None
            };
            let mut before = Vec::new();
            if v == lc.core_start {
                // Prelude snapshot: prelude_x, prelude blocks, then the
                // previous-pass bank and message base start as copies of them.
                before.push(CopyOp {
                    src: Slot::X(x),
                    dst: Slot::PreludeX,
                });
                for i in 0..np {
                    before.push(CopyOp {
                        src: Slot::Cur(i as u8),
                        dst: Slot::Pre(i as u8),
                    });
                }
                for i in 0..np {
                    before.push(CopyOp {
                        src: Slot::Pre(i as u8),
                        dst: Slot::Prev(i as u8),
                    });
                }
                before.push(CopyOp {
                    src: Slot::PreludeX,
                    dst: Slot::PrevOut,
                });
            }
            if let Some(p) = pass {
                if v == lc.core_start + p * core_len {
                    if p > 0 {
                        // End of pass p-1: its blocks and stream become "previous".
                        for i in 0..n_prev {
                            before.push(CopyOp {
                                src: Slot::Cur(i as u8),
                                dst: Slot::Prev(i as u8),
                            });
                        }
                        before.push(CopyOp {
                            src: Slot::X(x),
                            dst: Slot::PrevOut,
                        });
                    }
                    // Start of pass p: stream and blocks restart from the prelude.
                    before.push(CopyOp {
                        src: Slot::PreludeX,
                        dst: Slot::X(x),
                    });
                    for i in 0..np {
                        before.push(CopyOp {
                            src: Slot::Pre(i as u8),
                            dst: Slot::Cur(i as u8),
                        });
                    }
                    cur_len = np;
                }
            }
            let core_i = match pass {
                Some(p) if p > 0 => Some(((v - lc.core_start) % core_len) as u8),
                _ => None,
            };
            let push = (phys % block == 0).then_some(cur_len as u8);
            let desc = VisitDesc {
                visit: v as u8,
                phys: phys as u8,
                swa: cfg.swa_layers[phys],
                ring_rows: u32::try_from(visit_rows[v])
                    .map_err(|_| "inkling2 decode: ring rows exceed u32".to_string())?,
                phase: match pass {
                    Some(p) => Phase::Pass(p as u8),
                    None if v < lc.core_start => Phase::Prelude,
                    None => Phase::Post,
                },
                core_i,
                x_in: x,
                x_out: 1 - x,
                blocks_len: cur_len as u8,
                prev: core_i.map(|_| ((np + 1) as u8, n_prev as u8)),
                push,
                before,
            };
            if push.is_some() {
                cur_len += 1;
            }
            x = 1 - x;
            visits.push(desc);
        }
        if cur_len != n_cur {
            return Err(format!(
                "inkling2 decode: schedule ends with {cur_len} blocks, expected {n_cur}"
            ));
        }
        Ok(Self {
            visits,
            n_pre: n_pre as u8,
            n_prev: n_prev as u8,
            n_cur: n_cur as u8,
            x_final: x,
        })
    }

    /// Deterministic digest for logging (selection compares the schedules).
    pub fn digest(&self) -> u64 {
        let mut h = DefaultHasher::new();
        self.hash(&mut h);
        h.finish()
    }
}

// ---------------------------------------------------------------------------
// Buckets, key, plan
// ---------------------------------------------------------------------------

/// A live-context bucket: inclusive key-count interval `[lo, hi]` (`n_full =
/// pos + 1`) and the immutable FA2 grid width of every visit inside it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DecodeBucket {
    pub lo: u32,
    pub hi: u32,
    /// `max(splits(min(n, ring)))` over `n in [lo, hi]`, per visit.
    pub split_cap: [u16; MAX_VISITS],
}

/// All buckets of a load: cuts clipped to the position limit, deduplicated.
fn build_buckets(
    cfg: &Inkling2Config,
    max_seq: usize,
    visit_rows: &[usize],
) -> Result<Vec<DecodeBucket>, String> {
    let limit = cfg.gpu_position_limit(max_seq).min(max_seq);
    if limit == 0 {
        return Err("inkling2 decode: no executable position".into());
    }
    let mut cuts: Vec<usize> = BUCKET_CUTS.iter().map(|&c| c.min(limit)).collect();
    cuts.push(limit);
    cuts.sort_unstable();
    cuts.dedup();
    let mut out = Vec::with_capacity(cuts.len());
    let mut lo = 1usize;
    for hi in cuts {
        if hi < lo {
            continue;
        }
        let mut split_cap = [0u16; MAX_VISITS];
        let mut by_ring: Vec<(usize, u16)> = Vec::new();
        for (v, &ring) in visit_rows.iter().enumerate() {
            let cap = match by_ring.iter().find(|r| r.0 == ring) {
                Some(r) => r.1,
                None => {
                    let c = (lo..=hi)
                        .map(|n| inkling2_attn_fa2_splits(n.min(ring)))
                        .max()
                        .unwrap_or(0);
                    if c == 0 || c > INKLING2_ATTN_MAX_SPLITS {
                        return Err(format!("inkling2 decode: split cap {c} for ring {ring}"));
                    }
                    by_ring.push((ring, c as u16));
                    c as u16
                }
            };
            split_cap[v] = cap;
        }
        out.push(DecodeBucket {
            lo: lo as u32,
            hi: hi as u32,
            split_cap,
        });
        lo = hi + 1;
    }
    Ok(out)
}

/// Device identity a tape is bound to.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct DeviceIdentity {
    pub arch: String,
    pub device_id: i32,
}

/// Route-affecting recipe of the body.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DecodeRecipe {
    pub arch_id: u32,
    pub abi_rev: u32,
    pub kv_f16: bool,
    pub dense_tier: u8,
    pub expert_tier: u8,
    pub attn_fa2: bool,
    pub fused_producer: bool,
    pub route_row: bool,
}

/// Capacity half of the key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DecodeCapacity {
    pub max_seq: u32,
    pub position_limit: u32,
    pub arena_bytes: u64,
    pub fa2_max_splits: u32,
}

/// Exact selection key of a decode step (§3): a retained tape is valid only
/// for an equal key. `layout_epoch` follows the shared scratch growth epoch,
/// so any growth that could move a pointee changes the key.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DecodeStepKey {
    pub load_generation: u64,
    pub layout_epoch: u64,
    pub device: Arc<DeviceIdentity>,
    pub recipe: DecodeRecipe,
    pub schedule: Arc<DecodeSchedule>,
    pub capacity: DecodeCapacity,
    pub bucket: DecodeBucket,
}

/// A validated step: the key and, through it, the static visit/bucket
/// descriptor `enqueue_decode_body` consumes. No live `pos` / `n_kv`.
#[derive(Clone, Debug)]
pub struct DecodeStepPlan {
    pub key: DecodeStepKey,
}

fn tier_code(t: PrefillActTier) -> u8 {
    match t {
        PrefillActTier::F16 => 0,
        PrefillActTier::Fp8 => 1,
        PrefillActTier::A8 => 2,
        PrefillActTier::A4 => 3,
    }
}

static NEXT_LOAD_GENERATION: AtomicU64 = AtomicU64::new(1);

// ---------------------------------------------------------------------------
// Arena
// ---------------------------------------------------------------------------

/// Named byte region of the arena slab.
#[derive(Clone, Debug)]
pub struct ArenaRegion {
    pub name: &'static str,
    pub offset: usize,
    pub bytes: usize,
}

struct LayoutBuilder {
    end: usize,
    regions: Vec<ArenaRegion>,
}

impl LayoutBuilder {
    fn new() -> Self {
        Self {
            end: 0,
            regions: Vec::new(),
        }
    }
    fn add(&mut self, name: &'static str, bytes: usize) -> usize {
        let offset = self.end;
        self.regions.push(ArenaRegion {
            name,
            offset,
            bytes,
        });
        self.end = (offset + bytes).next_multiple_of(ALIGN);
        offset
    }
    /// Boundary and pairwise-liveness audit: aligned, in range, disjoint.
    fn finish(self) -> Result<(Vec<ArenaRegion>, usize), String> {
        for r in &self.regions {
            if r.offset % ALIGN != 0 || r.offset + r.bytes > self.end {
                return Err(format!("inkling2 decode: region {} misplaced", r.name));
            }
        }
        for (i, a) in self.regions.iter().enumerate() {
            for b in &self.regions[i + 1..] {
                if a.offset < b.offset + b.bytes && b.offset < a.offset + a.bytes {
                    return Err(format!(
                        "inkling2 decode: regions {} and {} overlap",
                        a.name, b.name
                    ));
                }
            }
        }
        Ok((self.regions, self.end))
    }
}

/// Aligned, range-checked typed view of the slab.
fn typed(slab: &GpuTensor, off: usize, shape: &[usize], dtype: DType) -> Result<GpuTensor, String> {
    let bytes = shape.iter().product::<usize>() * dtype.size();
    if off % ALIGN != 0 || off + bytes > slab.buf.size() {
        return Err(format!(
            "inkling2 decode: view [{off}, {}) outside the {} B arena",
            off + bytes,
            slab.buf.size()
        ));
    }
    Ok(GpuTensor {
        buf: slab.buf.byte_view(off, bytes),
        shape: shape.to_vec(),
        dtype,
    })
}

/// Typed views of one arena slab, before they are moved into the arena.
struct Views {
    io: GpuTensor,
    pos: GpuTensor,
    packed: GpuTensor,
    emb: GpuTensor,
    x: [GpuTensor; 2],
    prelude_x: GpuTensor,
    prev_out: GpuTensor,
    pre: Vec<GpuTensor>,
    prev: Vec<GpuTensor>,
    cur: Vec<GpuTensor>,
    hidden: GpuTensor,
    logits: GpuTensor,
    logits_row: GpuTensor,
    scratch: PrefillScratch,
    vis: GpuTensor,
}

/// Upper bound of the visit-temporary slab (the bump arena reports its true
/// high-water mark; exceeding this is a range error at the first token, never
/// a silent overrun).
fn vis_capacity(cfg: &Inkling2Config, m_total: usize) -> usize {
    let e = cfg.n_embd;
    let eattn = cfg.n_head * cfg.head_dim;
    let floats = 40 * e + 12 * eattn + 4 * m_total + 8 * cfg.loop_cfg.gate_rank + 8192;
    floats * 4 + 96 * ALIGN
}

/// The dedicated per-load M=1 arena of the frozen decode body.
///
/// One device allocation holds every stable operand: the staged input record
/// (`[pos:u32, pad16, packed row]`), the embedding output, the x ping-pong,
/// the prelude / previous / current pass banks, the head hidden row and
/// logits, M=1 slices of `x_rot` / `y_gu` / `y_dn` / `msg` (sized by
/// `PrefillScratch::sizes(cfg, 1)`), and the visit-temporary slab. The FA2
/// partition scratch is borrowed from the per-state buffers.
pub struct DecodeArena {
    slab: GpuTensor,
    regions: Vec<ArenaRegion>,
    host_rec: Vec<u8>,
    io: GpuTensor,
    pos: GpuTensor,
    packed: GpuTensor,
    emb: GpuTensor,
    x: [GpuTensor; 2],
    prelude_x: GpuTensor,
    prev_out: GpuTensor,
    pre: Vec<GpuTensor>,
    prev: Vec<GpuTensor>,
    cur: Vec<GpuTensor>,
    hidden: GpuTensor,
    logits: GpuTensor,
    logits_row: GpuTensor,
    scratch: PrefillScratch,
    fs: RefCell<FrozenScratch>,
    schedule: Arc<DecodeSchedule>,
    buckets: Vec<DecodeBucket>,
    device: Arc<DeviceIdentity>,
    capacity: DecodeCapacity,
    recipe: DecodeRecipe,
    load_generation: u64,
    vis_capacity: usize,
}

/// Entries of the three block banks of a schedule over `cfg`'s loop geometry.
fn bank_counts(cfg: &Inkling2Config) -> (usize, usize, usize) {
    let lc = &cfg.loop_cfg;
    let block = cfg.attn_res_block.max(1);
    let pushes = |r: std::ops::Range<usize>| r.filter(|p| p % block == 0).count();
    let (np, nc, npost) = (
        pushes(0..lc.core_start),
        pushes(lc.core_start..lc.core_end),
        pushes(lc.core_end..lc.n_phys),
    );
    (np, np + nc, np + nc + npost)
}

impl DecodeArena {
    /// Exact device bytes [`Self::new`] allocates for `cfg`.
    pub fn bytes_for(cfg: &Inkling2Config) -> Result<u64, String> {
        let (a, b, c) = bank_counts(cfg);
        let (_, bytes) = Self::layout(cfg, a, b, c)?;
        Ok(bytes as u64)
    }

    fn layout(
        cfg: &Inkling2Config,
        n_pre: usize,
        n_prev: usize,
        n_cur: usize,
    ) -> Result<(Vec<ArenaRegion>, usize), String> {
        let e = cfg.n_embd;
        let (f, _) = PrefillScratch::sizes(cfg, 1);
        let m_total = hipfire_dispatch::families::moe::grouped_moe_m_total_bound(
            cfg.n_expert_used,
            cfg.n_experts,
        );
        let row_bytes = e / 256 * 136;
        let mut lb = LayoutBuilder::new();
        lb.add("io", REC_HEAD + row_bytes);
        lb.add("emb", e * 4);
        lb.add("x0", e * 4);
        lb.add("x1", e * 4);
        lb.add("prelude_x", e * 4);
        lb.add("prev_out", e * 4);
        lb.add("hidden", e * 4);
        lb.add("pre", n_pre * e * 4);
        lb.add("prev", n_prev * e * 4);
        lb.add("cur", n_cur * e * 4);
        lb.add("logits", cfg.vocab_size * 4);
        lb.add("x_rot", f[0] * 4);
        lb.add("y_gu", f[1] * 4);
        lb.add("y_dn", f[2] * 4);
        lb.add("msg", f[3] * 4);
        lb.add("vis", vis_capacity(cfg, m_total));
        lb.finish()
    }

    /// Allocate the arena for this load. Allocation failure is returned (the
    /// caller keeps the stock eager decode); nothing is leaked.
    pub fn new(
        gpu: &mut Gpu,
        cfg: &Inkling2Config,
        gs: &Inkling2GpuState,
        embed_row_bytes: usize,
    ) -> Result<Self, String> {
        let e = cfg.n_embd;
        if embed_row_bytes != e / 256 * 136 {
            return Err(format!(
                "inkling2 decode: embed row {embed_row_bytes} B != {}",
                e / 256 * 136
            ));
        }
        if (e * 4) % ALIGN != 0 {
            return Err(format!(
                "inkling2 decode: n_embd {e} rows not 256 B aligned"
            ));
        }
        let schedule = Arc::new(DecodeSchedule::build(cfg, &gs.visit_rows)?);
        let buckets = build_buckets(cfg, gs.max_seq, &gs.visit_rows)?;
        let (n_pre, n_prev, n_cur) = (
            schedule.n_pre as usize,
            schedule.n_prev as usize,
            schedule.n_cur as usize,
        );
        let (regions, total) = Self::layout(cfg, n_pre, n_prev, n_cur)?;
        let slab = gpu
            .alloc_tensor(&[total], DType::Raw)
            .map_err(|e| format!("inkling2 decode: arena alloc ({total} B): {e:?}"))?;
        let parts = Self::views(cfg, &slab, &regions, n_pre, n_prev, n_cur, embed_row_bytes);
        let v = match parts {
            Ok(v) => v,
            Err(e) => {
                let _ = gpu.free_tensor(slab);
                return Err(e);
            }
        };
        let limit = cfg.gpu_position_limit(gs.max_seq).min(gs.max_seq);
        let routes = gs.prefill_routes;
        let capacity = DecodeCapacity {
            max_seq: gs.max_seq as u32,
            position_limit: limit as u32,
            arena_bytes: total as u64,
            fa2_max_splits: INKLING2_ATTN_MAX_SPLITS as u32,
        };
        let recipe = DecodeRecipe {
            arch_id: crate::config::ARCH_ID as u32,
            abi_rev: DECODE_ABI_REV,
            kv_f16: INKLING2_KV_DTYPE == DType::F16,
            dense_tier: tier_code(routes.dense),
            expert_tier: tier_code(routes.experts),
            attn_fa2: gs.attn_fa2,
            fused_producer: false,
            route_row: routes.experts == PrefillActTier::A8,
        };
        let vis_cap = v.vis.buf.size();
        Ok(Self {
            slab,
            regions,
            host_rec: vec![0u8; REC_HEAD + embed_row_bytes],
            io: v.io,
            pos: v.pos,
            packed: v.packed,
            emb: v.emb,
            x: v.x,
            prelude_x: v.prelude_x,
            prev_out: v.prev_out,
            pre: v.pre,
            prev: v.prev,
            cur: v.cur,
            hidden: v.hidden,
            logits: v.logits,
            logits_row: v.logits_row,
            scratch: v.scratch,
            fs: RefCell::new(FrozenScratch::over_slab(v.vis)),
            schedule,
            buckets,
            device: Arc::new(DeviceIdentity {
                arch: gpu.arch.clone(),
                device_id: gpu.device_id,
            }),
            capacity,
            recipe,
            load_generation: NEXT_LOAD_GENERATION.fetch_add(1, Ordering::Relaxed),
            vis_capacity: vis_cap,
        })
    }

    /// Typed, aligned, range-checked views of every region of `slab`.
    fn views(
        cfg: &Inkling2Config,
        slab: &GpuTensor,
        regions: &[ArenaRegion],
        n_pre: usize,
        n_prev: usize,
        n_cur: usize,
        embed_row_bytes: usize,
    ) -> Result<Views, String> {
        if slab.buf.as_ptr() as usize % ALIGN != 0 {
            return Err("inkling2 decode: arena base is not 256 B aligned".into());
        }
        let e = cfg.n_embd;
        let region = |name: &str| -> Result<&ArenaRegion, String> {
            regions
                .iter()
                .find(|r| r.name == name)
                .ok_or_else(|| format!("inkling2 decode: no region {name}"))
        };
        let off = |name: &str| -> Result<usize, String> { Ok(region(name)?.offset) };
        let f32row =
            |name: &str| -> Result<GpuTensor, String> { typed(slab, off(name)?, &[e], DType::F32) };
        let bank = |name: &str, n: usize| -> Result<Vec<GpuTensor>, String> {
            (0..n)
                .map(|i| typed(slab, off(name)? + i * e * 4, &[e], DType::F32))
                .collect()
        };
        let (f, _) = PrefillScratch::sizes(cfg, 1);
        let io = typed(slab, off("io")?, &[REC_HEAD + embed_row_bytes], DType::Raw)?;
        let pos = typed(slab, off("io")?, &[4], DType::Raw)?;
        let packed = GpuTensor {
            buf: slab.buf.byte_view(off("io")? + REC_HEAD, embed_row_bytes),
            shape: vec![embed_row_bytes],
            dtype: DType::Raw,
        };
        let scratch = PrefillScratch::decode_views(
            typed(slab, off("x_rot")?, &[f[0]], DType::F32)?,
            typed(slab, off("y_gu")?, &[f[1]], DType::F32)?,
            typed(slab, off("y_dn")?, &[f[2]], DType::F32)?,
            typed(slab, off("msg")?, &[f[3]], DType::F32)?,
            alias(&packed),
        );
        let vis = typed(slab, off("vis")?, &[region("vis")?.bytes], DType::Raw)?;
        Ok(Views {
            io,
            pos,
            packed,
            emb: typed(slab, off("emb")?, &[1, e], DType::F32)?,
            x: [f32row("x0")?, f32row("x1")?],
            prelude_x: f32row("prelude_x")?,
            prev_out: f32row("prev_out")?,
            pre: bank("pre", n_pre)?,
            prev: bank("prev", n_prev)?,
            cur: bank("cur", n_cur)?,
            hidden: f32row("hidden")?,
            logits: typed(slab, off("logits")?, &[cfg.vocab_size], DType::F32)?,
            logits_row: typed(slab, off("logits")?, &[cfg.unpadded_vocab_size], DType::F32)?,
            scratch,
            vis,
        })
    }

    /// Pre-prime the shared `Gpu::scratch` slots this body touches at their
    /// exact sizes (A8 reservations at the gate/up and down widths, the MQ sign
    /// tables, the head GEMV's activation scratch) and resolve the inline-kernel
    /// modules, so a step never grows a shared pointee or JITs a module first
    /// seen mid-body.
    pub fn prime(
        &self,
        gpu: &mut Gpu,
        cfg: &Inkling2Config,
        gw: &Inkling2GpuWeights,
    ) -> Result<(), String> {
        let m = |what: &str, r: Result<_, hip_bridge::HipError>| -> Result<_, String> {
            r.map_err(|e| format!("inkling2 decode: prime {what}: {e:?}"))
        };
        ensure_kernels(gpu)?;
        ensure_pf_kernels(gpu)?;
        m("mq signs", gpu.ensure_mq_signs())?;
        let slots = cfg.n_expert_used;
        let phys = crate::weights::xts_down_dims(cfg).1;
        m(
            "gate/up a8",
            gpu.reserve_int8_mmq(cfg.n_embd, slots).map(|_| ()),
        )?;
        m("down a8", gpu.reserve_int8_mmq(phys, slots).map(|_| ()))?;
        // Head: one GEMV on the arena's own hidden/logits rows sizes the
        // activation scratch exactly as the real head does.
        let hv = GpuTensor {
            buf: self.hidden.buf.byte_view(0, self.hidden.buf.size()),
            shape: vec![1, cfg.n_embd],
            dtype: DType::F32,
        };
        m("head", weight_gemv(gpu, &gw.unembed, &hv, &self.logits))?;
        Ok(())
    }

    /// Exact device bytes of this arena.
    pub fn bytes(&self) -> u64 {
        self.capacity.arena_bytes
    }

    /// The named regions, in layout order.
    pub fn regions(&self) -> &[ArenaRegion] {
        &self.regions
    }

    /// Fresh owner identity of this arena (new on every successful load/ensure).
    pub fn load_generation(&self) -> u64 {
        self.load_generation
    }

    /// Device base address of the slab (stable for the arena's life).
    pub fn base_ptr(&self) -> usize {
        self.slab.buf.as_ptr() as usize
    }

    pub fn schedule(&self) -> &Arc<DecodeSchedule> {
        &self.schedule
    }

    pub fn buckets(&self) -> &[DecodeBucket] {
        &self.buckets
    }

    /// Largest visit-temporary extent carved so far / its capacity (bytes).
    pub fn vis_high_water(&self) -> (usize, usize) {
        (self.fs.borrow().high_water(), self.vis_capacity)
    }

    /// Named stable tensors for the oracle (banks, ping-pong, head rows).
    pub fn named_tensors(&self) -> Vec<(String, &GpuTensor)> {
        let mut v: Vec<(String, &GpuTensor)> = vec![
            ("x0".into(), &self.x[0]),
            ("x1".into(), &self.x[1]),
            ("prelude_x".into(), &self.prelude_x),
            ("prev_out".into(), &self.prev_out),
            ("hidden".into(), &self.hidden),
            ("logits".into(), &self.logits),
        ];
        for (i, t) in self.pre.iter().enumerate() {
            v.push((format!("pre{i}"), t));
        }
        for (i, t) in self.prev.iter().enumerate() {
            v.push((format!("prev{i}"), t));
        }
        for (i, t) in self.cur.iter().enumerate() {
            v.push((format!("cur{i}"), t));
        }
        v
    }

    /// The whole slab (oracle poisoning / hashing only).
    pub fn slab(&self) -> &GpuTensor {
        &self.slab
    }

    fn slot(&self, s: Slot) -> &GpuTensor {
        match s {
            Slot::X(i) => &self.x[i as usize],
            Slot::PreludeX => &self.prelude_x,
            Slot::PrevOut => &self.prev_out,
            Slot::Pre(i) => &self.pre[i as usize],
            Slot::Prev(i) => &self.prev[i as usize],
            Slot::Cur(i) => &self.cur[i as usize],
        }
    }

    /// Release the allocation (the views alias it and die with the arena).
    pub fn free(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.slab);
    }
}

/// Non-owning clone of a tensor header over the same bytes.
fn alias(t: &GpuTensor) -> GpuTensor {
    GpuTensor {
        buf: t.buf.byte_view(0, t.buf.size()),
        shape: t.shape.clone(),
        dtype: t.dtype,
    }
}

/// Create the load's decode arena (idempotent) and prime the shared scratch.
/// Fails (leaving the load untouched) when the geometry or device does not run
/// the frozen body, or the allocation does not fit; the caller then keeps the
/// stock eager decode.
pub fn ensure_decode_arena(gpu: &mut Gpu, bundle: &mut Inkling2Bundle) -> Result<(), String> {
    if bundle
        .gpu_state
        .as_ref()
        .is_some_and(|s| s.decode_arena.is_some())
    {
        return Ok(());
    }
    if !prefill_supported(bundle, gpu) {
        return Err("inkling2 decode: the blocked one-row body is unsupported on this load".into());
    }
    let gw = bundle
        .gpu_weights
        .as_ref()
        .ok_or("inkling2 decode: no gpu weights")?;
    let gs = bundle
        .gpu_state
        .as_mut()
        .ok_or("inkling2 decode: no gpu state")?;
    if !gs.attn_fa2 {
        return Err("inkling2 decode: the frozen body needs the FA2 decode attention".into());
    }
    let arena = DecodeArena::new(gpu, &bundle.config, gs, gw.embed_packed.row_bytes())?;
    if let Err(e) = arena.prime(gpu, &bundle.config, gw) {
        arena.free(gpu);
        return Err(e);
    }
    gs.decode_arena = Some(arena);
    Ok(())
}

// ---------------------------------------------------------------------------
// plan / stage / enqueue / finish
// ---------------------------------------------------------------------------

/// Host validation of one step at `pos`, before anything is submitted:
/// `pos == seq_pos`, every visit's `visit_cap == pos`, the true position tags
/// of every key the step reads, `pos + 1 <= max_seq` and inside the unit-τ
/// position limit, and the load's arena. Returns the selection key.
pub fn plan_decode_step(
    bundle: &Inkling2Bundle,
    gpu: &Gpu,
    pos: usize,
) -> Result<DecodeStepPlan, String> {
    let gs = bundle
        .gpu_state
        .as_ref()
        .ok_or("inkling2 decode: no gpu state")?;
    let arena = gs
        .decode_arena
        .as_ref()
        .ok_or("inkling2 decode: this load has no decode arena (ensure_decode_arena)")?;
    if !prefill_supported(bundle, gpu) {
        return Err("inkling2 decode: the blocked one-row body is unsupported on this load".into());
    }
    if !gs.attn_fa2 {
        return Err("inkling2 decode: the frozen body needs the FA2 decode attention".into());
    }
    if pos != gs.seq_pos {
        return Err(format!(
            "inkling2 decode: pos {pos} != seq_pos {} (out-of-order)",
            gs.seq_pos
        ));
    }
    if pos + 1 > gs.max_seq {
        return Err(format!(
            "inkling2 decode: pos {pos} beyond max_seq {}",
            gs.max_seq
        ));
    }
    if pos >= arena.capacity.position_limit as usize {
        return Err(format!(
            "inkling2 decode: pos {pos} beyond the unit-tau position limit {}",
            arena.capacity.position_limit
        ));
    }
    for d in &arena.schedule.visits {
        let v = d.visit as usize;
        if gs.visit_cap[v] != pos {
            return Err(format!(
                "inkling2 decode: visit {v} cursor {} != pos {pos}",
                gs.visit_cap[v]
            ));
        }
        if !d.swa && pos + 1 > gs.visit_rows[v] {
            return Err(format!(
                "inkling2 decode: visit {v} KV overflow at pos {pos} (rows {})",
                gs.visit_rows[v]
            ));
        }
        check_pre_block_tags(gs, v, d.swa, pos)?;
    }
    let n_full = (pos + 1) as u32;
    let bi = arena.buckets.partition_point(|b| b.hi < n_full);
    let bucket = *arena
        .buckets
        .get(bi)
        .filter(|b| b.lo <= n_full)
        .ok_or_else(|| format!("inkling2 decode: no bucket for {n_full} keys"))?;
    let mut recipe = arena.recipe;
    recipe.fused_producer = gpu.flags.gfx12_producer_quant_fused_enabled();
    Ok(DecodeStepPlan {
        key: DecodeStepKey {
            load_generation: arena.load_generation,
            layout_epoch: rdna_compute::scratch::scratch_growth_epoch(),
            device: arena.device.clone(),
            recipe,
            schedule: arena.schedule.clone(),
            capacity: arena.capacity,
            bucket,
        },
    })
}

/// Stage one step's input: gather the token's packed row and the absolute
/// position into the arena's host record and issue the ONE ordered H2D into
/// the stable input allocation. The record is overwritten completely each
/// call (the padding is never read).
pub fn stage_decode_input(
    gpu: &mut Gpu,
    weights: &Inkling2GpuWeights,
    arena: &mut DecodeArena,
    token_id: u32,
    pos: u32,
) -> Result<(), String> {
    arena.host_rec[..4].copy_from_slice(&pos.to_ne_bytes());
    weights
        .embed_packed
        .copy_row(token_id, &mut arena.host_rec[REC_HEAD..])?;
    gpu.upload_stream_ordered(&arena.io.buf, &arena.host_rec)
        .map_err(|e| format!("inkling2 decode: stage input: {e:?}"))
}

/// Enqueue the body: embed unpack + norm, the 48 visits, one head + softcap.
/// Enqueue only: no allocation or release, H2D/D2H, synchronization, JIT-first
/// resolution of arena state, or host cursor / tag mutation. `Ok` is NOT a
/// commit; [`finish_decode_step`] commits after the logits are read back.
pub fn enqueue_decode_body(
    gpu: &mut Gpu,
    cfg: &Inkling2Config,
    weights: &Inkling2GpuWeights,
    state: &Inkling2GpuState,
    arena: &DecodeArena,
    step: &DecodeStepPlan,
) -> Result<(), String> {
    run_body(gpu, cfg, weights, state, arena, step, None)
}

/// [`enqueue_decode_body`] with per-visit `visit_out` downloads (the stock
/// trace order). Diagnostic only: it synchronizes after every visit.
pub fn enqueue_decode_body_trace(
    gpu: &mut Gpu,
    cfg: &Inkling2Config,
    weights: &Inkling2GpuWeights,
    state: &Inkling2GpuState,
    arena: &DecodeArena,
    step: &DecodeStepPlan,
    trace: &mut Vec<Vec<f32>>,
) -> Result<(), String> {
    trace.clear();
    run_body(gpu, cfg, weights, state, arena, step, Some(trace))
}

fn run_body(
    gpu: &mut Gpu,
    cfg: &Inkling2Config,
    weights: &Inkling2GpuWeights,
    state: &Inkling2GpuState,
    arena: &DecodeArena,
    step: &DecodeStepPlan,
    trace: Option<&mut Vec<Vec<f32>>>,
) -> Result<(), String> {
    let k = &step.key;
    if k.load_generation != arena.load_generation
        || k.schedule != arena.schedule
        || k.capacity != arena.capacity
    {
        return Err("inkling2 decode: stale step plan for this arena".into());
    }
    let routes = state.prefill_routes;
    let scope = frozen_scope_enter(gpu, routes);
    let r = body(gpu, cfg, weights, state, arena, &k.bucket, routes, trace);
    frozen_scope_exit(gpu, scope);
    r
}

#[allow(clippy::too_many_arguments)]
fn body(
    gpu: &mut Gpu,
    cfg: &Inkling2Config,
    weights: &Inkling2GpuWeights,
    state: &Inkling2GpuState,
    arena: &DecodeArena,
    bucket: &DecodeBucket,
    routes: crate::gpu_prefill::PrefillRoutes,
    mut trace: Option<&mut Vec<Vec<f32>>>,
) -> Result<(), String> {
    let e = cfg.n_embd;
    let hip = |what: &str, r: Result<(), hip_bridge::HipError>| {
        r.map_err(|er| format!("inkling2 decode: {what}: {er:?}"))
    };
    if weights.layers.len() != cfg.n_phys {
        return Err(format!(
            "inkling2 decode: {} layers != n_phys {}",
            weights.layers.len(),
            cfg.n_phys
        ));
    }
    if weights.cross_pass.len() != cfg.loop_cfg.core_end - cfg.loop_cfg.core_start {
        return Err("inkling2 decode: cross_pass length != core_len".into());
    }
    // Embed unpack (device) and token RMSNorm into x[0].
    hip(
        "embed decode",
        gpu.inkling2_embed_qt44_rows(&arena.packed, &arena.emb, 1, e),
    )?;
    hip(
        "embed norm",
        gpu.rmsnorm_batched(
            &arena.emb,
            &weights.embed_norm,
            &arena.x[0],
            1,
            e,
            cfg.rms_norm_eps,
        ),
    )?;
    let mut fs = arena
        .fs
        .try_borrow_mut()
        .map_err(|_| "inkling2 decode: arena scratch already in use".to_string())?;
    for d in &arena.schedule.visits {
        let vi = d.visit as usize;
        for c in &d.before {
            frozen_copy(gpu, arena.slot(c.src), arena.slot(c.dst), e)?;
        }
        let f = FrozenVisit {
            pos: &arena.pos,
            split_cap: bucket.split_cap[vi] as usize,
            visit_out: &arena.x[d.x_out as usize],
            push_dst: d.push.map(|i| &arena.cur[i as usize]),
        };
        let msgs: Option<Msgs<'_>> = match (d.core_i, d.prev) {
            (Some(i), Some((skip, len))) => {
                let ctrls = weights.layers[d.phys as usize]
                    .controllers
                    .as_ref()
                    .ok_or_else(|| {
                        format!("inkling2 decode: core layer {} missing controllers", d.phys)
                    })?;
                Some((
                    &arena.prev[skip as usize..len as usize],
                    &arena.prev_out,
                    &weights.cross_pass[i as usize],
                    ctrls,
                ))
            }
            _ => None,
        };
        frozen_visit(
            gpu,
            cfg,
            weights,
            state,
            &arena.scratch,
            routes,
            &mut fs,
            &f,
            &arena.x[d.x_in as usize],
            &arena.cur[..d.blocks_len as usize],
            vi,
            msgs,
        )?;
        if let Some(t) = trace.as_mut() {
            let row = gpu
                .download_f32(&arena.x[d.x_out as usize])
                .map_err(|er| format!("inkling2 decode: trace: {er:?}"))?;
            t.push(row);
        }
    }
    frozen_head(
        gpu,
        cfg,
        weights,
        &arena.cur[..arena.schedule.n_cur as usize],
        &arena.x[arena.schedule.x_final as usize],
        &arena.hidden,
        &arena.logits,
    )
}

/// Complete a step: read the unpadded logits back (the ONE D2H) and only then
/// commit every visit's position tag and cursor and `seq_pos`.
pub fn finish_decode_step(
    gpu: &Gpu,
    state: &mut Inkling2GpuState,
    arena: &DecodeArena,
    pos: usize,
    logits: &mut Vec<f32>,
) -> Result<(), String> {
    if state.seq_pos != pos {
        return Err(format!(
            "inkling2 decode: finish at pos {pos} but seq_pos is {}",
            state.seq_pos
        ));
    }
    // On an owned stream (the graph route) the readback is ordered on it: one
    // wait, no legacy-stream hop (the blocking copy makes the runtime join the
    // null queue with the decode queue, which can time-slice every dispatch of
    // the step). With no stream it is the blocking copy, unchanged.
    let read = if gpu.active_stream.is_some() {
        logits.clear();
        gpu.download_f32_stream_ordered_append(&mut [(&arena.logits_row, &mut *logits)])
    } else {
        gpu.download_f32_into(&arena.logits_row, logits)
    };
    read.map_err(|e| format!("inkling2 decode: logits download: {e:?}"))?;
    for d in &arena.schedule.visits {
        let v = d.visit as usize;
        let ring = state.visit_rows[v];
        state.visit_pos_tags[v][pos % ring] = pos as u32;
        state.visit_cap[v] = pos + 1;
    }
    state.seq_pos = pos + 1;
    Ok(())
}

/// The frozen eager decode: plan, stage, enqueue, finish on the ordinary
/// stream. Bitwise the incumbent [`crate::forward_decode_gpu`] (the oracle),
/// with its logits written into the caller's reusable `Vec`. Creates the load's
/// arena on first use.
pub fn forward_decode_frozen(
    bundle: &mut Inkling2Bundle,
    gpu: &mut Gpu,
    token_id: u32,
    pos: usize,
    logits: &mut Vec<f32>,
) -> Result<(), String> {
    ensure_decode_arena(gpu, bundle)?;
    let plan = plan_decode_step(bundle, gpu, pos)?;
    let pos32 = u32::try_from(pos).map_err(|_| "inkling2 decode: pos exceeds u32".to_string())?;
    let cfg = &bundle.config;
    let gw = bundle
        .gpu_weights
        .as_ref()
        .ok_or("inkling2 decode: no gpu weights")?;
    let gs = bundle
        .gpu_state
        .as_mut()
        .ok_or("inkling2 decode: no gpu state")?;
    let mut arena = gs
        .decode_arena
        .take()
        .ok_or("inkling2 decode: no decode arena")?;
    let r = (|| {
        stage_decode_input(gpu, gw, &mut arena, token_id, pos32)?;
        enqueue_decode_body(gpu, cfg, gw, gs, &arena, &plan)?;
        finish_decode_step(gpu, gs, &arena, pos, logits)
    })();
    gs.decode_arena = Some(arena);
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Inkling2Config {
        Inkling2Config::from_metadata_json(crate::TEST_METADATA).expect("config")
    }

    fn rings(cfg: &Inkling2Config, max_seq: usize) -> Vec<usize> {
        (0..cfg.loop_cfg.n_slot())
            .map(|v| {
                if cfg.swa_layers[cfg.loop_cfg.layer_phys(v)] {
                    max_seq.min(crate::gpu::SWA_WINDOW_ROWS)
                } else {
                    max_seq
                }
            })
            .collect()
    }

    /// The static schedule performs exactly the stock driver's bank traffic:
    /// 18 bank copies (4 prelude, 3 x 2 pass starts, 2 x 4 pass ends) and the
    /// four block pushes, in the loop order.
    #[test]
    fn schedule_mirrors_the_stock_driver() {
        let c = cfg();
        let s = DecodeSchedule::build(&c, &rings(&c, 2053)).unwrap();
        assert_eq!(s.visits.len(), 48);
        assert_eq!((s.n_pre, s.n_prev, s.n_cur), (1, 3, 4));
        let copies: usize = s.visits.iter().map(|d| d.before.len()).sum();
        assert_eq!(copies, 18);
        let pushes: Vec<usize> = s
            .visits
            .iter()
            .filter(|d| d.push.is_some())
            .map(|d| d.visit as usize)
            .collect();
        // phys 0, 6, 12 (pass 0), then 6 and 12 again in passes 1 and 2, then 18.
        assert_eq!(pushes, vec![0, 6, 12, 18, 24, 30, 36, 42]);
        for d in &s.visits {
            assert_ne!(d.x_in, d.x_out);
            assert_eq!(
                d.core_i.is_some(),
                matches!(d.phase, Phase::Pass(p) if p > 0)
            );
            assert_eq!(d.prev.is_some(), d.core_i.is_some());
            if let Some(p) = d.prev {
                assert_eq!(p, (2, 3));
            }
            assert_eq!(d.phys as usize, c.loop_cfg.layer_phys(d.visit as usize));
        }
        // Block-bank length at visit start: 1 in the prelude after its push,
        // restarting at 1 each pass, 3 in the post layers.
        let len = |v: usize| s.visits[v].blocks_len;
        assert_eq!(
            (
                len(0),
                len(1),
                len(6),
                len(12),
                len(18),
                len(30),
                len(42),
                len(47)
            ),
            (0, 1, 1, 2, 1, 1, 3, 4)
        );
        assert_eq!(s.x_final, 0);
        assert_eq!(s, DecodeSchedule::build(&c, &rings(&c, 2053)).unwrap());
        assert_ne!(
            s.digest(),
            DecodeSchedule::build(&c, &rings(&c, 4100))
                .unwrap()
                .digest(),
            "capacity changes the ring rows and so the schedule"
        );
    }

    /// Buckets: cuts at 128 / 192 / 2048 / 4096 clipped to the position limit,
    /// the limit as the final bound; per-visit FA2 grid width from the actual
    /// split rule (SWA clamps at the ring).
    #[test]
    fn buckets_follow_the_split_rule() {
        let c = cfg();
        let r = rings(&c, 8192);
        let swa = (0..48)
            .find(|&v| c.swa_layers[c.loop_cfg.layer_phys(v)])
            .unwrap();
        let full = (0..48)
            .find(|&v| !c.swa_layers[c.loop_cfg.layer_phys(v)])
            .unwrap();
        let b = build_buckets(&c, 8192, &r).unwrap();
        let shape: Vec<(u32, u32, u16, u16)> = b
            .iter()
            .map(|k| (k.lo, k.hi, k.split_cap[full], k.split_cap[swa]))
            .collect();
        assert_eq!(
            shape,
            vec![
                (1, 128, 2, 2),
                (129, 192, 4, 4),
                (193, 2048, 8, 8),
                (2049, 4096, 16, 8),
                (4097, 8192, 8, 8),
            ]
        );
        // Clipped capacity: the cuts clip, the limit closes the last bucket.
        let r = rings(&c, 2053);
        let b = build_buckets(&c, 2053, &r).unwrap();
        let bounds: Vec<(u32, u32)> = b.iter().map(|k| (k.lo, k.hi)).collect();
        assert_eq!(
            bounds,
            vec![(1, 128), (129, 192), (193, 2048), (2049, 2053)]
        );
        assert_eq!(b[3].split_cap[full], 16);
        // A cap on the 1024 ring break: SWA ring == full rows.
        let b = build_buckets(&c, 1024, &rings(&c, 1024)).unwrap();
        assert_eq!(b.last().map(|k| (k.lo, k.hi)), Some((193, 1024)));
        // Smaller than the first cut.
        let b = build_buckets(&c, 100, &rings(&c, 100)).unwrap();
        assert_eq!(b.len(), 1);
        assert_eq!((b[0].lo, b[0].hi, b[0].split_cap[full]), (1, 100, 2));
        // The limit of an over-long capacity is the first non-unit-tau position.
        let b = build_buckets(&c, 32768, &rings(&c, 32768)).unwrap();
        assert_eq!(
            b.last().map(|k| k.hi),
            Some(c.gpu_position_limit(32768) as u32)
        );
        // Every key count maps to exactly one bucket.
        for n in [1u32, 128, 129, 192, 193, 2048, 2049, 4096, 4097] {
            let hits = b.iter().filter(|k| k.lo <= n && n <= k.hi).count();
            assert_eq!(hits, 1, "n_full {n}");
        }
    }

    #[test]
    fn arena_layout_is_aligned_disjoint_and_exact() {
        let c = cfg();
        let (a, b, d) = bank_counts(&c);
        assert_eq!((a, b, d), (1, 3, 4));
        let (regions, total) = DecodeArena::layout(&c, a, b, d).unwrap();
        assert!(regions.iter().all(|r| r.offset % ALIGN == 0));
        assert_eq!(total % ALIGN, 0);
        assert_eq!(DecodeArena::bytes_for(&c).unwrap(), total as u64);
        let io = regions.iter().find(|r| r.name == "io").unwrap();
        assert_eq!(io.bytes, REC_HEAD + c.n_embd / 256 * 136);
    }
}
