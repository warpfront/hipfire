// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! TP2 trunk shard of the Flash-Next EP2 forward (`HIPFIRE_QWEN4_EP_TRUNK_SHARD`: default on for
//! an EP2 load over two gfx1201 cards, `0` is the kill switch; Gate 0b F-R shape,
//! `mgmtp/trunkshard/DESIGN.md`).
//!
//! Each rank keeps half of the replicated trunk projections, cut as byte ranges from the
//! resident (replicated, unchanged-artifact) tensors, exactly like
//! [`Qwen4Weights::shard_dense_ffn`]: download, slice, upload, [`replace_resident`] frees the
//! full copy. The cuts are format-agnostic (`MatrixGeometry` reads the tensor's dtype: a row
//! slice is whole rows, a column slice is whole quant groups), so an overlay that already
//! replaced a tensor (Q8_0, MQ8) is sliced in its own format; run this after
//! [`Qwen4Weights::apply_trunk_overlay`] and before the first forward and before the expert
//! plan so the census sees the freed bytes ([`trunk_share_bytes`]).
//!
//! What is cut (rank `r` of `parts`, Flash-Next 16 key / 48 value GDN heads of 128, 24 q /
//! 2 KV QSA heads of 256):
//! * GDN `in_proj_qkv` / `conv1d` rows `q[r] | k[r] | v[r]` (key heads and the value heads
//!   of their groups: value head `h` belongs to key head `h / (vh / kh)`), `in_proj_z` rows,
//!   `in_proj_a` / `in_proj_b` / `A_log` / `dt_bias` rows by value head, `out_proj` columns
//!   by value head.
//! * QSA `q_proj` rows are PER-HEAD interleaved on the device (`[head][q 256 | gate 256]`,
//!   the attention kernels read `q_with_gate + head * 2 * head_dim`; the CPU reference's
//!   `[Q rows | gate rows]` block layout is not the device's):
//!   [`QsaMode::GroupInterleaved`] gives rank `r` heads `g * (qh / kvh) + r * hpr ..` of
//!   every KV group `g` (`hpr = qh / kvh / parts`), so a rank runs `qh / parts` q heads over
//!   the SAME `kvh` KV heads at a GQA ratio `hpr` with k/v projections and the KV cache
//!   replicated (no 1-KV-head attention or even-KV-head fp8 format needed); `o_proj` columns
//!   follow the same head order. [`QsaMode::KvSplit`] is the Gate 0b shape (contiguous q
//!   heads, KV head `r`), valid only when `kvh % parts == 0`.
//! * shared expert `gate`/`up` rows and `down` columns in 128-wide units (640 = 384 + 256),
//!   `lm_head` rows (vocab).
//! Everything else (HC, router, norms, indexer, embedding, PLE, MTP head, routed experts)
//! stays replicated.

use super::*;
use rdna_compute::DType as Dt;

/// Developer switch of the trunk shard.
pub const TRUNK_SHARD_ENV: &str = "HIPFIRE_QWEN4_EP_TRUNK_SHARD";

/// Whether the `lm_head` is vocab-row sharded. It needs the per-rank top-k / argmax exchange
/// (slice S3c: logits scratch, argmax, sampler and MTP candidate sites all assume full vocab
/// rows), so until then it stays replicated and is neither cut nor counted by the census.
pub const SHARD_LM_HEAD: bool = false;

/// Shared-expert intermediate granule: the prefill kernels admit `intermediate % 128`.
pub const SHARED_UNIT: usize = 128;

/// What the raw `HIPFIRE_QWEN4_EP_TRUNK_SHARD` value asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShardSwitch {
    /// Unset (or empty): the shipped default, on for an EP2 load over two gfx1201 cards.
    Default,
    /// `0` / `off` / `false`: the kill switch, EP2 as before the shard.
    Off,
    /// `1` / `true` / `on`: the shard even where the default would not apply (a load that is
    /// not two-rank EP refuses, loudly, in [`Qwen4Bundle::apply_trunk_shard`]).
    On,
}

/// Parse the raw switch. An unrecognised value is OFF (fail safe to the pre-shard route).
pub fn parse_switch(raw: Option<&str>) -> ShardSwitch {
    match raw.map(str::trim) {
        None | Some("") => ShardSwitch::Default,
        Some("1" | "true" | "on") => ShardSwitch::On,
        Some(_) => ShardSwitch::Off,
    }
}

/// The raw environment switch ([`parse_switch`] of `HIPFIRE_QWEN4_EP_TRUNK_SHARD`).
pub fn trunk_shard_switch() -> ShardSwitch {
    parse_switch(
        hipfire_config::developer_var(TRUNK_SHARD_ENV)
            .ok()
            .as_deref(),
    )
}

/// Whether the shipped default applies: expert parallelism of exactly two ranks over exactly two
/// visible gfx1201 cards (the only shape the fused all-reduce kernel and the measurements cover).
pub fn default_applies_given(ep: usize, archs: &[&str]) -> bool {
    ep == 2 && archs.len() == 2 && archs.iter().all(|arch| *arch == "gfx1201")
}

/// [`default_applies_given`] for this process: the Qwen4 expert-parallel degree
/// (`topology.ep`, `HIPFIRE_QWEN4_EP=0` forces 1) and the visible GPUs. A process that cannot
/// name its GPUs (no readable KFD topology) gets no default.
pub fn default_applies() -> bool {
    use std::sync::OnceLock;
    static ARCHS: OnceLock<Option<Vec<String>>> = OnceLock::new();
    let archs = ARCHS.get_or_init(|| {
        hipfire_config::devices::startup_devices()
            .map(|devices| devices.iter().map(|d| d.arch.clone()).collect())
    });
    let Some(archs) = archs else {
        return false;
    };
    let archs: Vec<&str> = archs.iter().map(String::as_str).collect();
    default_applies_given(hipfire_config::topology::qwen4_ep().unwrap_or(1), &archs)
}

/// [`trunk_shard_requested`] with both inputs passed in (the CPU tests of the load path drive
/// every `*_for(requested)` helper below with explicit values instead of the process environment).
pub const fn requested_for(switch: ShardSwitch, default_applies: bool) -> bool {
    match switch {
        ShardSwitch::Off => false,
        ShardSwitch::On => true,
        ShardSwitch::Default => default_applies,
    }
}

/// Whether this load shards the trunk: the kill switch, the explicit `1`, or the default.
pub fn trunk_shard_requested() -> bool {
    match trunk_shard_switch() {
        ShardSwitch::Off => false,
        ShardSwitch::On => true,
        ShardSwitch::Default => default_applies(),
    }
}

/// `HIPFIRE_QWEN4_EP_TRUNK_SHARD_MTP=0` (developer): drop the native MTP head under the shard
/// (AR-only load: the head's weights and expert units go back to the trunk). Unset keeps the
/// head, replicated on every rank.
pub const TRUNK_SHARD_MTP_ENV: &str = "HIPFIRE_QWEN4_EP_TRUNK_SHARD_MTP";

/// The raw `HIPFIRE_QWEN4_EP_TRUNK_SHARD_MTP` switch (default on: keep the head).
pub fn trunk_shard_mtp_env_on() -> bool {
    hipfire_config::developer_bool(TRUNK_SHARD_MTP_ENV, true)
}

/// [`trunk_shard_keeps_mtp`] with both inputs passed in: whether the shard is requested and the
/// MTP switch. Both are required: without the shard the head policy is not the shard's.
pub const fn keeps_mtp_given(requested: bool, mtp_env: bool) -> bool {
    requested && mtp_env
}

/// Whether the native MTP head stays loaded under the shard: replicated on every rank (its
/// weights are not cut) and run on the GLOBAL geometry ([`Qwen4Config::mtp_geometry`]: 24 q
/// heads, shared expert 640) while the 48-layer trunk runs sharded on the rank-local config.
/// Default on whenever the shard is; `HIPFIRE_QWEN4_EP_TRUNK_SHARD_MTP=0` drops it. THE one
/// decision for it: artifact admission ([`crate::artifact`]) keeps the profile / `mtp` config /
/// `mtp.*` manifest or drops them ([`Qwen4Config::into_headless`]) from it, and
/// [`load_local_config`] refuses a config that disagrees; every spec/lane gate follows from
/// the admitted config.
pub fn trunk_shard_keeps_mtp() -> bool {
    keeps_mtp_given(trunk_shard_requested(), trunk_shard_mtp_env_on())
}

/// Whether a load drops the native MTP head because the shard asks for it.
pub fn trunk_shard_drops_mtp_head() -> bool {
    drops_head_for(trunk_shard_requested(), trunk_shard_keeps_mtp())
}

/// [`trunk_shard_drops_mtp_head`] for an explicit `requested` and `keeps`
/// ([`trunk_shard_keeps_mtp`]).
pub const fn drops_head_for(requested: bool, keeps: bool) -> bool {
    requested && !keeps
}

/// The census view of the shard for a `ranks`-way EP load: `Some` when the switch is on and
/// the load is two-rank (the only shard width), else `None` (unchanged census). `config` is
/// the GLOBAL config (the census sizes the whole tensors the shard cuts).
pub fn census_trunk(config: &Qwen4Config, ranks: usize) -> Option<(TrunkDims, QsaMode)> {
    census_trunk_for(trunk_shard_requested(), config, ranks)
}

/// [`census_trunk`] for an explicit `requested`.
pub fn census_trunk_for(
    requested: bool,
    config: &Qwen4Config,
    ranks: usize,
) -> Option<(TrunkDims, QsaMode)> {
    (ranks == 2 && requested).then(|| {
        // A rank-local config would halve the dims twice: always cut from the global one.
        let global = config.replica_geometry();
        (TrunkDims::from_config(&global), QsaMode::GroupInterleaved)
    })
}

/// The rank-local config a two-rank EP load of rank `placement.rank` runs under when the
/// shard is on (`Ok(None)` when it is off). Refused loudly when the switch is on but the
/// load is not two-rank EP, or when the config carries the native MTP head the policy drops
/// (artifact admission follows [`trunk_shard_drops_mtp_head`], so this is a defence). A kept
/// head rides on the local config as its global [`Qwen4Config::mtp_geometry`].
pub fn load_local_config(
    config: &Qwen4Config,
    placement: Option<Qwen4EpPlacement>,
) -> Result<Option<Qwen4Config>, String> {
    load_local_config_for(
        trunk_shard_requested(),
        trunk_shard_keeps_mtp(),
        config,
        placement,
    )
}

/// [`load_local_config`] for an explicit `requested` and `keeps` ([`trunk_shard_keeps_mtp`]).
pub fn load_local_config_for(
    requested: bool,
    keeps: bool,
    config: &Qwen4Config,
    placement: Option<Qwen4EpPlacement>,
) -> Result<Option<Qwen4Config>, String> {
    if !requested {
        return Ok(None);
    }
    if config.mtp.is_some() && !keeps {
        return Err(format!(
            "{TRUNK_SHARD_ENV}: the config carries the native MTP head but the shard drops it \
             (unset {TRUNK_SHARD_MTP_ENV}, or set it to 1, to keep it replicated); \
             artifact admission should have admitted the artifact headless"
        ));
    }
    let placement = placement
        .filter(|p| p.ranks == 2)
        .ok_or_else(|| format!("{TRUNK_SHARD_ENV} needs a two-rank expert-parallel load"))?;
    config
        .trunk_shard_local(placement.rank, placement.ranks)
        .map(Some)
}

/// How the QSA layers are split.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QsaMode {
    /// Each rank runs `qh / parts` q heads, half of every KV group, over all KV heads
    /// (k/v projections and KV cache replicated).
    GroupInterleaved,
    /// Contiguous q heads and KV head `r` (needs `kvh % parts == 0` and a 1-KV-head
    /// attention/KV format).
    KvSplit,
}

/// One cut of a row-major `[rows, cols]` matrix (a vector is `[n, 1]`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Cut {
    /// Whole rows, concatenated in range order.
    Rows(Vec<Range<usize>>),
    /// Whole quant groups of every row (column ranges in input units), concatenated per row.
    Cols(Vec<Range<usize>>),
    /// Kept whole on every rank.
    Replicate,
}

impl Cut {
    fn total(ranges: &[Range<usize>]) -> usize {
        ranges.iter().map(|r| r.end - r.start).sum()
    }

    /// Shape of the cut of a `[rows, cols]` matrix.
    pub fn out_shape(&self, rows: usize, cols: usize) -> [usize; 2] {
        match self {
            Cut::Rows(r) => [Self::total(r), cols],
            Cut::Cols(c) => [rows, Self::total(c)],
            Cut::Replicate => [rows, cols],
        }
    }
}

/// The tensors of the trunk shard.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Role {
    GdnQkv,
    GdnConv,
    GdnZ,
    GdnA,
    GdnB,
    GdnALog,
    GdnDtBias,
    GdnOut,
    QsaQ,
    QsaK,
    QsaV,
    QsaO,
    SharedGate,
    SharedUp,
    SharedDown,
    LmHead,
}

/// Model dimensions the cuts depend on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrunkDims {
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub shared_inter: usize,
    pub vocab: usize,
}

impl TrunkDims {
    pub fn from_config(config: &Qwen4Config) -> Self {
        Self {
            key_heads: config.linear_num_key_heads,
            value_heads: config.linear_num_value_heads,
            key_dim: config.linear_key_head_dim,
            value_dim: config.linear_value_head_dim,
            q_heads: config.num_attention_heads,
            kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            shared_inter: config.shared_expert_intermediate_size,
            vocab: config.vocab_size,
        }
    }

    fn check(&self, parts: usize) -> Result<(), String> {
        if parts < 2 {
            return Err(format!("trunk shard needs two or more parts, got {parts}"));
        }
        for (what, n) in [
            ("linear key heads", self.key_heads),
            ("linear value heads", self.value_heads),
            ("q heads", self.q_heads),
        ] {
            if n % parts != 0 {
                return Err(format!("{what} {n} are not divisible by {parts} parts"));
            }
        }
        if self.key_heads == 0 || self.value_heads % self.key_heads != 0 {
            return Err(format!(
                "value heads {} are not a multiple of key heads {}",
                self.value_heads, self.key_heads
            ));
        }
        if self.kv_heads == 0 || self.q_heads % self.kv_heads != 0 {
            return Err(format!(
                "q heads {} are not a multiple of KV heads {}",
                self.q_heads, self.kv_heads
            ));
        }
        Ok(())
    }
}

fn balanced(rank: usize, parts: usize, total: usize) -> Range<usize> {
    hipfire_runtime::tp_shard::ShardConfig::balanced_range(rank, parts, total)
}

/// Cut of `role` for `rank` of `parts`; `None` for a tensor this role never touches.
pub fn role_cut(
    d: &TrunkDims,
    mode: QsaMode,
    role: Role,
    rank: usize,
    parts: usize,
) -> Result<Cut, String> {
    d.check(parts)?;
    if rank >= parts {
        return Err(format!("rank {rank} outside {parts} parts"));
    }
    let (kw, vw) = (d.key_heads * d.key_dim, d.value_heads * d.value_dim);
    let (kper, vper) = (kw / parts, vw / parts);
    let vh_per = d.value_heads / parts;
    let qw = d.q_heads * d.head_dim;
    let cut = match role {
        Role::GdnQkv | Role::GdnConv => Cut::Rows(vec![
            rank * kper..(rank + 1) * kper,
            kw + rank * kper..kw + (rank + 1) * kper,
            2 * kw + rank * vper..2 * kw + (rank + 1) * vper,
        ]),
        Role::GdnZ => Cut::Rows(vec![rank * vper..(rank + 1) * vper]),
        Role::GdnA | Role::GdnB | Role::GdnALog | Role::GdnDtBias => {
            Cut::Rows(vec![rank * vh_per..(rank + 1) * vh_per])
        }
        Role::GdnOut => Cut::Cols(vec![rank * vper..(rank + 1) * vper]),
        // The device `q_proj` rows are PER-HEAD interleaved (`[head][q | gate]`, 512 rows
        // per head: `head_stride_rows: 512`, the attention kernels read `q_with_gate + head *
        // 2 * head_dim` for q and `+ head_dim` for the gate), so a kept head is ONE 2 * head_dim
        // row range. (The CPU reference's `[Q rows | gate rows]` block layout is not the
        // device's.)
        Role::QsaQ => match mode {
            QsaMode::GroupInterleaved => Cut::Rows(
                qsa_head_ranges(d, rank, parts)?
                    .into_iter()
                    .map(|r| 2 * r.start..2 * r.end)
                    .collect(),
            ),
            QsaMode::KvSplit => {
                let per = qw / parts;
                Cut::Rows(vec![2 * rank * per..2 * (rank + 1) * per])
            }
        },
        Role::QsaO => match mode {
            QsaMode::GroupInterleaved => Cut::Cols(qsa_head_ranges(d, rank, parts)?),
            QsaMode::KvSplit => {
                let per = qw / parts;
                Cut::Cols(vec![rank * per..(rank + 1) * per])
            }
        },
        Role::QsaK | Role::QsaV => match mode {
            QsaMode::GroupInterleaved => Cut::Replicate,
            QsaMode::KvSplit => {
                if d.kv_heads % parts != 0 {
                    return Err(format!(
                        "KV split needs kv heads {} % {parts} == 0",
                        d.kv_heads
                    ));
                }
                let per = d.kv_heads / parts * d.head_dim;
                Cut::Rows(vec![rank * per..(rank + 1) * per])
            }
        },
        Role::SharedGate | Role::SharedUp | Role::SharedDown => {
            if d.shared_inter % SHARED_UNIT != 0 {
                return Err(format!(
                    "shared intermediate {} is not a multiple of {SHARED_UNIT}",
                    d.shared_inter
                ));
            }
            let units = balanced(rank, parts, d.shared_inter / SHARED_UNIT);
            let range = units.start * SHARED_UNIT..units.end * SHARED_UNIT;
            if range.is_empty() {
                return Err(format!(
                    "shared intermediate {} cannot feed {parts} parts",
                    d.shared_inter
                ));
            }
            if role == Role::SharedDown {
                Cut::Cols(vec![range])
            } else {
                Cut::Rows(vec![range])
            }
        }
        Role::LmHead => Cut::Rows(vec![balanced(rank, parts, d.vocab)]),
    };
    Ok(cut)
}

/// Element ranges (in units of `head_dim`) of the q heads rank `rank` keeps in
/// [`QsaMode::GroupInterleaved`]: heads `g * group + rank * hpr .. + hpr` of every KV group.
pub fn qsa_head_ranges(
    d: &TrunkDims,
    rank: usize,
    parts: usize,
) -> Result<Vec<Range<usize>>, String> {
    let group = d.q_heads / d.kv_heads;
    if group % parts != 0 {
        return Err(format!(
            "GQA group {group} is not divisible by {parts} parts (use KvSplit)"
        ));
    }
    let hpr = group / parts;
    Ok((0..d.kv_heads)
        .map(|g| {
            let first = g * group + rank * hpr;
            first * d.head_dim..(first + hpr) * d.head_dim
        })
        .collect())
}

// ----------------------------------------------------------------------------------------
// byte cuts
// ----------------------------------------------------------------------------------------

/// Group geometry of a stored matrix format: `group_k` input columns per group,
/// `group_bytes` bytes per group (rows are `ceil(cols / group_k)` groups).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatrixGeometry {
    pub group_k: usize,
    pub group_bytes: usize,
}

/// Geometry of the formats the shard can cut; `None` for anything else (Lloyd codebooks with
/// per-tensor LUT sidecars, MQ4G128V2 with row-local rotation tiles, blocked/tiled layouts).
pub fn matrix_geometry(dtype: Dt) -> Option<MatrixGeometry> {
    let group_k = match dtype {
        Dt::F32 | Dt::F16 | Dt::BF16 => 1,
        Dt::Q8_0 => 32,
        Dt::MQ4G256V2 | Dt::MQ5G256V2 | Dt::MQ6G256V2 | Dt::MQ8G256 => 256,
        _ => return None,
    };
    let group_bytes = dtype.row_bytes(group_k)?;
    Some(MatrixGeometry {
        group_k,
        group_bytes,
    })
}

/// Bytes of one row of `cols` input columns, or an error if `cols` is not whole groups.
fn row_bytes(g: MatrixGeometry, cols: usize) -> Result<usize, String> {
    if cols == 0 || cols % g.group_k != 0 {
        return Err(format!(
            "{cols} columns are not whole groups of {}",
            g.group_k
        ));
    }
    Ok(cols / g.group_k * g.group_bytes)
}

/// Apply `cut` to the bytes of a row-major `[rows, cols]` matrix of `geometry`. Row cuts need
/// only uniform rows; column cuts need group-aligned ranges. Returns the bytes and the shape.
pub fn cut_matrix(
    data: &[u8],
    geometry: MatrixGeometry,
    rows: usize,
    cols: usize,
    cut: &Cut,
) -> Result<(Vec<u8>, [usize; 2]), String> {
    let rb = row_bytes(geometry, cols)?;
    if data.len() != rows * rb {
        return Err(format!(
            "{} bytes is not a {rows}x{cols} matrix of {rb}-byte rows (blocked or padded layout?)",
            data.len()
        ));
    }
    let shape = cut.out_shape(rows, cols);
    let out = match cut {
        Cut::Replicate => data.to_vec(),
        Cut::Rows(ranges) => {
            let mut out = Vec::with_capacity(Cut::total(ranges) * rb);
            for r in ranges {
                if r.start > r.end || r.end > rows {
                    return Err(format!("row range {r:?} outside {rows} rows"));
                }
                out.extend_from_slice(&data[r.start * rb..r.end * rb]);
            }
            out
        }
        Cut::Cols(ranges) => {
            for r in ranges {
                if r.start > r.end
                    || r.end > cols
                    || r.start % geometry.group_k != 0
                    || r.end % geometry.group_k != 0
                {
                    return Err(format!(
                        "column range {r:?} is not group aligned ({}) inside {cols}",
                        geometry.group_k
                    ));
                }
            }
            let mut out = Vec::with_capacity(
                rows * Cut::total(ranges) / geometry.group_k * geometry.group_bytes,
            );
            for row in 0..rows {
                for r in ranges {
                    let a = row * rb + r.start / geometry.group_k * geometry.group_bytes;
                    let b = row * rb + r.end / geometry.group_k * geometry.group_bytes;
                    out.extend_from_slice(&data[a..b]);
                }
            }
            out
        }
    };
    Ok((out, shape))
}

/// Row cut of an array whose rows are `data.len() / rows` bytes of any element type (the
/// vector and depthwise-conv tensors: `A_log`, `dt_bias`, `conv1d [n,1,4]`).
pub fn cut_rows_raw(data: &[u8], rows: usize, ranges: &[Range<usize>]) -> Result<Vec<u8>, String> {
    if rows == 0 || data.len() % rows != 0 {
        return Err(format!("{} bytes is not {rows} uniform rows", data.len()));
    }
    let rb = data.len() / rows;
    let mut out = Vec::with_capacity(Cut::total(ranges) * rb);
    for r in ranges {
        if r.start > r.end || r.end > rows {
            return Err(format!("row range {r:?} outside {rows} rows"));
        }
        out.extend_from_slice(&data[r.start * rb..r.end * rb]);
    }
    Ok(out)
}

// ----------------------------------------------------------------------------------------
// census
// ----------------------------------------------------------------------------------------

/// The role of a manifest tensor name, if the shard cuts it.
pub fn role_of_name(name: &str) -> Option<Role> {
    if name == QWEN4_LM_HEAD {
        return SHARD_LM_HEAD.then_some(Role::LmHead);
    }
    if name.starts_with("mtp.") {
        return None; // the MTP head stays replicated
    }
    let tail = |s: &str| name.ends_with(s);
    Some(match () {
        _ if tail(".linear_attn.in_proj_qkv.weight") => Role::GdnQkv,
        _ if tail(".linear_attn.conv1d.weight") => Role::GdnConv,
        _ if tail(".linear_attn.in_proj_z.weight") => Role::GdnZ,
        _ if tail(".linear_attn.in_proj_a.weight") => Role::GdnA,
        _ if tail(".linear_attn.in_proj_b.weight") => Role::GdnB,
        _ if tail(".linear_attn.A_log") => Role::GdnALog,
        _ if tail(".linear_attn.dt_bias") => Role::GdnDtBias,
        _ if tail(".linear_attn.out_proj.weight") => Role::GdnOut,
        _ if tail(".self_attn.q_proj.weight") => Role::QsaQ,
        _ if tail(".self_attn.k_proj.weight") => Role::QsaK,
        _ if tail(".self_attn.v_proj.weight") => Role::QsaV,
        _ if tail(".self_attn.o_proj.weight") => Role::QsaO,
        _ if tail(".shared_expert.gate_proj.weight") => Role::SharedGate,
        _ if tail(".shared_expert.up_proj.weight") => Role::SharedUp,
        _ if tail(".shared_expert.down_proj.weight") => Role::SharedDown,
        _ => return None,
    })
}

/// Device bytes of one resident tensor on the BUSIEST rank of a `parts`-way trunk shard, from
/// its full `bytes` and logical `[rows, cols]` shape (`rows` for vectors, `[n, 1]`): the
/// kept fraction of rows (row cuts) or columns (column cuts) rounded up, so a budget built on
/// it never over-commits a rank. A tensor the shard does not cut (or `parts < 2`) is unchanged.
pub fn trunk_share_bytes(
    d: &TrunkDims,
    mode: QsaMode,
    name: &str,
    shape: &[usize],
    bytes: u64,
    parts: usize,
) -> Result<u64, String> {
    let Some(role) = role_of_name(name) else {
        return Ok(bytes);
    };
    if parts < 2 {
        return Ok(bytes);
    }
    let (rows, cols) = match shape {
        [r] => (*r, 1usize),
        [r, c, ..] => (*r, *c),
        [] => return Err(format!("{name}: empty shape")),
    };
    let mut worst = 0u64;
    for rank in 0..parts {
        let cut = role_cut(d, mode, role, rank, parts)?;
        let (num, den) = match &cut {
            Cut::Replicate => (1usize, 1usize),
            Cut::Rows(r) => (Cut::total(r), rows),
            Cut::Cols(c) => (Cut::total(c), cols),
        };
        if den == 0 || num > den {
            return Err(format!("{name}: cut {cut:?} does not fit shape {shape:?}"));
        }
        let share = (u128::from(bytes) * num as u128).div_ceil(den as u128) as u64;
        worst = worst.max(share);
    }
    Ok(worst)
}

// ----------------------------------------------------------------------------------------
// the loader step
// ----------------------------------------------------------------------------------------

/// What [`Qwen4Weights::shard_trunk`] did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrunkShardApplied {
    pub tensors: usize,
    pub before_bytes: usize,
    pub after_bytes: usize,
}

/// The first `bytes` bytes of a resident tensor's device buffer (no `byte_size` guess).
fn download_exact(gpu: &mut Gpu, tensor: &GpuTensor, bytes: usize) -> Result<Vec<u8>, String> {
    gpu.bind_thread().map_err(|e| e.to_string())?;
    let mut data = vec![0u8; bytes];
    gpu.hip
        .memcpy_dtoh(&mut data, &tensor.buf)
        .map_err(|e| e.to_string())?;
    gpu.take_device_wait_error().map_err(|e| e.to_string())?;
    Ok(data)
}

impl Qwen4Weights {
    /// Whether [`Self::shard_trunk`] ran.
    pub fn trunk_sharded(&self) -> Option<(usize, usize)> {
        self.trunk_shard
    }

    fn slot_of(&self, reference: &TensorRef) -> Result<usize, String> {
        self.taken_by_name
            .get(&reference.name)
            .into_iter()
            .flatten()
            .copied()
            .find(|&slot| {
                let key = &self.taken[slot].key;
                key.layer == reference.layer && key.device == self.device
            })
            .ok_or_else(|| format!("{}: no resident handle", reference.name))
    }

    /// Cut one resident tensor: returns `(slot, bytes, new logical shape, dtype)` or `None` for
    /// a replicated cut.
    fn trunk_cut_one(
        &self,
        gpu: &mut Gpu,
        reference: &TensorRef,
        cut: &Cut,
        raw_rows: bool,
    ) -> Result<Option<(usize, Vec<u8>, Vec<usize>, Dt)>, String> {
        if matches!(cut, Cut::Replicate) {
            return Ok(None);
        }
        let context = &reference.name;
        let tensor = self.resident(reference).map_err(|e| e.to_string())?;
        if tensor.buf.is_borrowed() {
            return Err(format!(
                "{context}: arena-backed resident cannot shrink in place"
            ));
        }
        let dtype = tensor.dtype;
        // The exact payload bytes, from the logical shape and the format's geometry: a
        // quantized resident carries its logical `[m, k]` shape, so `GpuTensor::byte_size`
        // (numel x `dtype.size()`) is NOT its device extent (MQ6: 1 B/elt vs 0.78 B/elt).
        let expected = if raw_rows {
            reference
                .shape
                .iter()
                .product::<usize>()
                .checked_mul(dtype.size())
        } else {
            match (reference.shape.as_slice(), matrix_geometry(dtype)) {
                ([rows, cols], Some(g)) => row_bytes(g, *cols)
                    .ok()
                    .and_then(|row| row.checked_mul(*rows)),
                _ => None,
            }
        }
        .ok_or_else(|| {
            format!(
                "{context}: no payload extent for {dtype:?} of logical shape {:?}",
                reference.shape
            )
        })?;
        if expected > tensor.buf.size() {
            return Err(format!(
                "{context}: payload {expected} B exceeds the device buffer of {} B",
                tensor.buf.size()
            ));
        }
        let bytes = download_exact(gpu, tensor, expected)?;
        if raw_rows {
            // vector / depthwise-conv tensors: rows of any element type
            let rows = reference.shape[0];
            let Cut::Rows(ranges) = cut else {
                return Err(format!("{context}: a raw tensor only takes a row cut"));
            };
            let out = cut_rows_raw(&bytes, rows, ranges)?;
            let mut shape = reference.shape.clone();
            shape[0] = Cut::total(ranges);
            return Ok(Some((self.slot_of(reference)?, out, shape, dtype)));
        }
        let [rows, cols] = reference.shape[..] else {
            return Err(format!(
                "{context}: shape {:?} is not a matrix",
                reference.shape
            ));
        };
        let geometry =
            matrix_geometry(dtype).ok_or_else(|| format!("{context}: {dtype:?} cannot be cut"))?;
        let (out, shape) = cut_matrix(&bytes, geometry, rows, cols, cut)?;
        Ok(Some((self.slot_of(reference)?, out, shape.to_vec(), dtype)))
    }

    /// Keep only TP rank `rank` of `parts` of the trunk projections (module docs). Run after any
    /// load-time replacement (requant, trunk overlay) and before the first forward.
    /// Every tensor of a layer is cut before any is installed.
    pub fn shard_trunk(
        &mut self,
        gpu: &mut Gpu,
        config: &Qwen4Config,
        mode: QsaMode,
        rank: usize,
        parts: usize,
    ) -> Result<TrunkShardApplied, String> {
        if self.trunk_shard.is_some() || self.tp_shard.is_some() {
            return Err(format!(
                "trunk shard: weights already sharded (trunk {:?}, dense ffn {:?})",
                self.trunk_shard, self.tp_shard
            ));
        }
        let dims = TrunkDims::from_config(config);
        dims.check(parts)?;
        if rank >= parts {
            return Err(format!("trunk shard rank {rank} outside {parts} parts"));
        }
        let (mut tensors, mut before, mut after) = (0usize, 0usize, 0usize);
        let mut install = |this: &mut Self,
                           gpu: &mut Gpu,
                           cuts: Vec<(usize, Vec<u8>, Vec<usize>, Dt)>|
         -> Result<Vec<Vec<usize>>, String> {
            let mut shapes = Vec::with_capacity(cuts.len());
            for (slot, bytes, shape, dtype) in cuts {
                let upload_shape: Vec<usize> =
                    if dtype == Dt::Q8_0 || matrix_geometry(dtype).is_some_and(|g| g.group_k > 1) {
                        vec![bytes.len()]
                    } else {
                        shape.clone()
                    };
                let mut tensor = gpu
                    .upload_raw(&bytes, &upload_shape)
                    .map_err(|e| e.to_string())?;
                tensor.dtype = dtype;
                let (old, new) =
                    replace_resident(gpu, &mut this.taken[slot], Replacement::Tensor(tensor))?;
                before += old;
                after += new;
                tensors += 1;
                shapes.push(shape);
            }
            Ok(shapes)
        };
        for index in 0..self.layer_refs.len() {
            // plan: (role, reference clone, raw_rows)
            let layer = self.layer_refs[index].clone();
            let mut plan: Vec<(Role, TensorRef, bool)> = Vec::new();
            if let Some(g) = &layer.gdn {
                plan.push((Role::GdnQkv, g.qkv.clone(), false));
                plan.push((Role::GdnZ, g.z.clone(), false));
                plan.push((Role::GdnA, g.in_proj_a.clone(), false));
                plan.push((Role::GdnB, g.in_proj_b.clone(), false));
                plan.push((Role::GdnConv, g.conv.clone(), true));
                plan.push((Role::GdnALog, g.a_log.clone(), true));
                plan.push((Role::GdnDtBias, g.dt_bias.clone(), true));
                plan.push((Role::GdnOut, g.output.clone(), false));
            }
            if let Some(a) = &layer.attention {
                plan.push((Role::QsaQ, a.q.clone(), false));
                plan.push((Role::QsaK, a.k.clone(), false));
                plan.push((Role::QsaV, a.v.clone(), false));
                plan.push((Role::QsaO, a.output.clone(), false));
            }
            if let Qwen4FfnWeights::Moe(m) = &layer.ffn {
                plan.push((Role::SharedGate, m.shared_gate.clone(), false));
                plan.push((Role::SharedUp, m.shared_up.clone(), false));
                plan.push((Role::SharedDown, m.shared_down.clone(), false));
            }
            let mut cuts = Vec::new();
            let mut roles = Vec::new();
            for (role, reference, raw) in &plan {
                let cut = role_cut(&dims, mode, *role, rank, parts)?;
                if let Some(one) = self.trunk_cut_one(gpu, reference, &cut, *raw)? {
                    cuts.push(one);
                    roles.push(*role);
                }
            }
            let shapes = install(self, gpu, cuts)?;
            let l = &mut self.layer_refs[index];
            for (role, shape) in roles.into_iter().zip(shapes) {
                let target: Option<&mut TensorRef> = match role {
                    Role::GdnQkv => l.gdn.as_mut().map(|g| &mut g.qkv),
                    Role::GdnZ => l.gdn.as_mut().map(|g| &mut g.z),
                    Role::GdnA => l.gdn.as_mut().map(|g| &mut g.in_proj_a),
                    Role::GdnB => l.gdn.as_mut().map(|g| &mut g.in_proj_b),
                    Role::GdnConv => l.gdn.as_mut().map(|g| &mut g.conv),
                    Role::GdnALog => l.gdn.as_mut().map(|g| &mut g.a_log),
                    Role::GdnDtBias => l.gdn.as_mut().map(|g| &mut g.dt_bias),
                    Role::GdnOut => l.gdn.as_mut().map(|g| &mut g.output),
                    Role::QsaQ => l.attention.as_mut().map(|a| &mut a.q),
                    Role::QsaK => l.attention.as_mut().map(|a| &mut a.k),
                    Role::QsaV => l.attention.as_mut().map(|a| &mut a.v),
                    Role::QsaO => l.attention.as_mut().map(|a| &mut a.output),
                    Role::SharedGate | Role::SharedUp | Role::SharedDown => match &mut l.ffn {
                        Qwen4FfnWeights::Moe(m) => Some(match role {
                            Role::SharedGate => &mut m.shared_gate,
                            Role::SharedUp => &mut m.shared_up,
                            _ => &mut m.shared_down,
                        }),
                        Qwen4FfnWeights::Dense(_) => None,
                    },
                    Role::LmHead => None,
                };
                if let Some(reference) = target {
                    reference.shape = shape;
                }
            }
        }
        if SHARD_LM_HEAD {
            let head = self.root.lm_head.clone();
            let cut = role_cut(&dims, mode, Role::LmHead, rank, parts)?;
            if let Some(one) = self.trunk_cut_one(gpu, &head, &cut, false)? {
                let shapes = install(self, gpu, vec![one])?;
                self.root.lm_head.shape = shapes.into_iter().next().expect("one shape");
            }
        }
        self.trunk_shard = Some((rank, parts));
        Ok(TrunkShardApplied {
            tensors,
            before_bytes: before,
            after_bytes: after,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FN: TrunkDims = TrunkDims {
        key_heads: 16,
        value_heads: 48,
        key_dim: 128,
        value_dim: 128,
        q_heads: 24,
        kv_heads: 2,
        head_dim: 256,
        shared_inter: 640,
        vocab: 248_320,
    };

    fn covered(cuts: &[Cut], rows: usize) -> Vec<u8> {
        let mut seen = vec![0u8; rows];
        for c in cuts {
            let Cut::Rows(rs) = c else {
                panic!("row cut expected")
            };
            for r in rs {
                for i in r.clone() {
                    seen[i] += 1;
                }
            }
        }
        seen
    }

    fn all_cuts(role: Role, mode: QsaMode) -> Vec<Cut> {
        (0..2)
            .map(|r| role_cut(&FN, mode, role, r, 2).unwrap())
            .collect()
    }

    /// Every row of the GDN/QSA/shared/lm_head row cuts is kept by exactly one rank.
    #[test]
    fn row_cuts_tile_exactly_once() {
        for (role, rows) in [
            (Role::GdnQkv, 10240),
            (Role::GdnConv, 10240),
            (Role::GdnZ, 6144),
            (Role::GdnA, 48),
            (Role::GdnB, 48),
            (Role::GdnALog, 48),
            (Role::GdnDtBias, 48),
            (Role::SharedGate, 640),
            (Role::SharedUp, 640),
            (Role::LmHead, 248_320),
        ] {
            let seen = covered(&all_cuts(role, QsaMode::GroupInterleaved), rows);
            assert!(seen.iter().all(|&c| c == 1), "{role:?}");
        }
        for mode in [QsaMode::GroupInterleaved, QsaMode::KvSplit] {
            let seen = covered(&all_cuts(Role::QsaQ, mode), 12288);
            assert!(seen.iter().all(|&c| c == 1), "QsaQ {mode:?}");
        }
        let seen = covered(&all_cuts(Role::QsaK, QsaMode::KvSplit), 512);
        assert!(seen.iter().all(|&c| c == 1));
    }

    /// Column cuts tile the K axis exactly once, in group-legal units.
    #[test]
    fn col_cuts_tile_exactly_once() {
        for (role, cols, mode) in [
            (Role::GdnOut, 6144, QsaMode::GroupInterleaved),
            (Role::QsaO, 6144, QsaMode::GroupInterleaved),
            (Role::QsaO, 6144, QsaMode::KvSplit),
            (Role::SharedDown, 640, QsaMode::GroupInterleaved),
        ] {
            let mut seen = vec![0u8; cols];
            for rank in 0..2 {
                let Cut::Cols(rs) = role_cut(&FN, mode, role, rank, 2).unwrap() else {
                    panic!()
                };
                for r in rs {
                    for i in r {
                        seen[i] += 1;
                    }
                }
            }
            assert!(seen.iter().all(|&c| c == 1), "{role:?}");
        }
        // Every column cut is a multiple of 128 so it is legal for MQ6/MQ8 (256-groups need
        // 256: checked per tensor in `cut_matrix`) and Q8_0/BF16 alike; the 3072 cuts are
        // 256-aligned, the shared 384/256 cut is 128-aligned (BF16 only).
        for role in [Role::GdnOut, Role::QsaO] {
            for rank in 0..2 {
                let Cut::Cols(rs) =
                    role_cut(&FN, QsaMode::GroupInterleaved, role, rank, 2).unwrap()
                else {
                    panic!()
                };
                assert!(
                    rs.iter().all(|r| r.start % 256 == 0 && r.end % 256 == 0),
                    "{role:?} {rs:?}"
                );
            }
        }
    }

    /// The GDN value head -> key head mapping survives the cut: rank r's local value head j
    /// (global `24r + j`) belongs to the local key head `j / 3` (global `8r + j / 3`), the same
    /// group the unsplit model uses, and the qkv/conv q|k|v blocks are in head order.
    #[test]
    fn gdn_head_grouping_is_preserved() {
        let ratio = FN.value_heads / FN.key_heads;
        for rank in 0..2 {
            let Cut::Rows(q) =
                role_cut(&FN, QsaMode::GroupInterleaved, Role::GdnQkv, rank, 2).unwrap()
            else {
                panic!()
            };
            assert_eq!(q[0], rank * 1024..(rank + 1) * 1024);
            assert_eq!(q[1], 2048 + rank * 1024..2048 + (rank + 1) * 1024);
            assert_eq!(q[2], 4096 + rank * 3072..4096 + (rank + 1) * 3072);
            let Cut::Rows(a) =
                role_cut(&FN, QsaMode::GroupInterleaved, Role::GdnA, rank, 2).unwrap()
            else {
                panic!()
            };
            let vh = a[0].clone();
            for (j, gv) in vh.clone().enumerate() {
                let global_key = gv / ratio;
                let local_key = j / ratio;
                assert_eq!(global_key, rank * (FN.key_heads / 2) + local_key);
            }
            // the v rows of the qkv cut are exactly those heads
            assert_eq!(q[2], 4096 + vh.start * 128..4096 + vh.end * 128);
        }
    }

    /// QSA GroupInterleaved: local q head `j` of rank r maps to local KV head `j / hpr`, which
    /// is the KV head of its global q head; o_proj columns follow the local head order; the
    /// `q_proj` rows are the same heads' per-head `[q | gate]` row pairs (2 * head_dim each).
    #[test]
    fn qsa_group_interleaved_keeps_the_gqa_mapping() {
        let group = FN.q_heads / FN.kv_heads; // 12
        let hpr = group / 2; // 6
        for rank in 0..2 {
            let Cut::Rows(rows) =
                role_cut(&FN, QsaMode::GroupInterleaved, Role::QsaQ, rank, 2).unwrap()
            else {
                panic!()
            };
            let Cut::Cols(cols) =
                role_cut(&FN, QsaMode::GroupInterleaved, Role::QsaO, rank, 2).unwrap()
            else {
                panic!()
            };
            assert_eq!(rows.len(), 2);
            for (r, c) in rows.iter().zip(&cols) {
                assert_eq!((r.start, r.end), (2 * c.start, 2 * c.end));
            }
            let mut local = 0usize;
            for c in &cols {
                for gh in c.start / 256..c.end / 256 {
                    assert_eq!(
                        gh / group,
                        local / hpr,
                        "rank {rank} local head {local} global {gh}"
                    );
                    local += 1;
                }
            }
            assert_eq!(local, FN.q_heads / 2);
        }
        assert_eq!(
            role_cut(&FN, QsaMode::GroupInterleaved, Role::QsaK, 0, 2).unwrap(),
            Cut::Replicate
        );
    }

    /// Shared expert: 640 = 384 + 256 in 128-wide units, rank 0 the larger.
    #[test]
    fn shared_split_is_384_256() {
        let Cut::Rows(r0) =
            role_cut(&FN, QsaMode::GroupInterleaved, Role::SharedGate, 0, 2).unwrap()
        else {
            panic!()
        };
        let Cut::Rows(r1) =
            role_cut(&FN, QsaMode::GroupInterleaved, Role::SharedGate, 1, 2).unwrap()
        else {
            panic!()
        };
        assert_eq!((r0[0].clone(), r1[0].clone()), (0..384, 384..640));
    }

    fn matrix(rows: usize, cols: usize, g: MatrixGeometry, seed: u8) -> Vec<u8> {
        let n = rows * cols / g.group_k * g.group_bytes;
        (0..n)
            .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
            .collect()
    }

    /// Byte cuts equal a reference cut done group by group, for every supported format, and a
    /// column cut then a row cut commutes with the row cut then the column cut.
    #[test]
    fn cut_matrix_matches_group_reference() {
        for dtype in [Dt::BF16, Dt::Q8_0, Dt::MQ6G256V2, Dt::MQ4G256V2] {
            let g = matrix_geometry(dtype).unwrap_or_else(|| panic!("{dtype:?}"));
            let (rows, cols) = (9usize, g.group_k.max(1) * 12);
            let data = matrix(rows, cols, g, 3);
            let groups = cols / g.group_k;
            let rb = groups * g.group_bytes;
            let grp = |row: usize, k: usize| {
                &data[row * rb + k * g.group_bytes..row * rb + (k + 1) * g.group_bytes]
            };
            let rcut = Cut::Rows(vec![2..4, 7..9]);
            let (out, shape) = cut_matrix(&data, g, rows, cols, &rcut).unwrap();
            assert_eq!(shape, [4, cols]);
            let want: Vec<u8> = [2usize, 3, 7, 8]
                .iter()
                .flat_map(|&r| data[r * rb..(r + 1) * rb].to_vec())
                .collect();
            assert_eq!(out, want);
            let c0 = g.group_k * 2;
            let c1 = g.group_k * 5;
            let c2 = g.group_k * 9;
            let c3 = g.group_k * 11;
            let ccut = Cut::Cols(vec![c0..c1, c2..c3]);
            let (cout, cshape) = cut_matrix(&data, g, rows, cols, &ccut).unwrap();
            assert_eq!(cshape, [rows, c1 - c0 + c3 - c2]);
            let mut want = Vec::new();
            for row in 0..rows {
                for k in (2..5).chain(9..11) {
                    want.extend_from_slice(grp(row, k));
                }
            }
            assert_eq!(cout, want);
            // commute
            let (a, sa) = cut_matrix(&out, g, 4, cols, &ccut).unwrap();
            let (rc, _) = cut_matrix(&cout, g, rows, sa[1], &rcut).unwrap();
            assert_eq!(a, rc);
        }
    }

    #[test]
    fn cut_matrix_refuses_misaligned_or_blocked() {
        let g = matrix_geometry(Dt::MQ6G256V2).unwrap();
        let data = matrix(4, 512, g, 1);
        assert!(cut_matrix(&data, g, 4, 512, &Cut::Cols(vec![0..128])).is_err());
        assert!(cut_matrix(&data[..data.len() - 1], g, 4, 512, &Cut::Rows(vec![0..1])).is_err());
        assert!(cut_matrix(&data, g, 4, 500, &Cut::Rows(vec![0..1])).is_err());
        assert!(matrix_geometry(Dt::MQ4G128V2).is_none());
        assert!(matrix_geometry(Dt::MQ4G256V2Lloyd).is_none());
    }

    /// The linear algebra the shard relies on: with a plain row-wise dequantization (Q8_0:
    /// f16 scale and 32 i8 per group; the rotation of the MQ formats is per 256-group, so it
    /// commutes with a group-aligned K split), concat of the row-cut GEMVs equals the full
    /// GEMV, and the sum of the K-cut partial GEMVs equals the full GEMV (exact in f64 up to
    /// reassociation).
    #[test]
    fn row_and_k_cuts_reproduce_the_gemv() {
        fn f16_bits(x: f32) -> u16 {
            // tiny scales are exact powers of two here
            let e = x.log2().round() as i32;
            (((e + 15) as u16) << 10) as u16
        }
        fn f16_val(b: u16) -> f64 {
            2f64.powi(i32::from(b >> 10) - 15)
        }
        let (rows, cols) = (6usize, 128usize);
        let g = matrix_geometry(Dt::Q8_0).unwrap();
        let mut data = Vec::new();
        let mut seed = 12345u32;
        let mut next = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed >> 8
        };
        for _ in 0..rows * cols / 32 {
            let scale = f16_bits(2f32.powi(-(next() as i32 % 4) - 3));
            data.extend_from_slice(&scale.to_le_bytes());
            for _ in 0..32 {
                data.push((next() % 255) as u8);
            }
        }
        let x: Vec<f64> = (0..cols)
            .map(|i| ((i * 7 % 13) as f64 - 6.0) / 8.0)
            .collect();
        let dequant = |bytes: &[u8], rows: usize, cols: usize| -> Vec<Vec<f64>> {
            let rb = cols / 32 * 34;
            (0..rows)
                .map(|r| {
                    let mut row = Vec::new();
                    for b in 0..cols / 32 {
                        let o = r * rb + b * 34;
                        let s = f16_val(u16::from_le_bytes([bytes[o], bytes[o + 1]]));
                        row.extend(bytes[o + 2..o + 34].iter().map(|&q| s * f64::from(q as i8)));
                    }
                    row
                })
                .collect()
        };
        let gemv = |w: &[Vec<f64>], x: &[f64]| -> Vec<f64> {
            w.iter()
                .map(|r| r.iter().zip(x).map(|(a, b)| a * b).sum())
                .collect()
        };
        let full = gemv(&dequant(&data, rows, cols), &x);
        // row cut
        let (r0, _) = cut_matrix(&data, g, rows, cols, &Cut::Rows(vec![0..3])).unwrap();
        let (r1, _) = cut_matrix(&data, g, rows, cols, &Cut::Rows(vec![3..6])).unwrap();
        let mut cat = gemv(&dequant(&r0, 3, cols), &x);
        cat.extend(gemv(&dequant(&r1, 3, cols), &x));
        assert_eq!(cat, full);
        // K cut
        let (k0, _) = cut_matrix(&data, g, rows, cols, &Cut::Cols(vec![0..64])).unwrap();
        let (k1, _) = cut_matrix(&data, g, rows, cols, &Cut::Cols(vec![64..128])).unwrap();
        let p0 = gemv(&dequant(&k0, rows, 64), &x[..64]);
        let p1 = gemv(&dequant(&k1, rows, 64), &x[64..]);
        for i in 0..rows {
            assert!((p0[i] + p1[i] - full[i]).abs() < 1e-9, "row {i}");
        }
    }

    #[test]
    fn raw_row_cut_takes_vectors_and_conv() {
        let conv: Vec<u8> = (0..10240 * 8).map(|i| i as u8).collect(); // [10240,1,4] bf16
        let Cut::Rows(r) = role_cut(&FN, QsaMode::GroupInterleaved, Role::GdnConv, 1, 2).unwrap()
        else {
            panic!()
        };
        let out = cut_rows_raw(&conv, 10240, &r).unwrap();
        assert_eq!(out.len(), 5120 * 8);
        assert_eq!(&out[..8], &conv[1024 * 8..1025 * 8]);
        let alog: Vec<u8> = (0..48 * 4).map(|i| i as u8).collect();
        let Cut::Rows(r) = role_cut(&FN, QsaMode::GroupInterleaved, Role::GdnALog, 1, 2).unwrap()
        else {
            panic!()
        };
        assert_eq!(
            cut_rows_raw(&alog, 48, &r).unwrap(),
            alog[24 * 4..].to_vec()
        );
    }

    #[test]
    fn names_map_to_roles() {
        let p = "model.language_model.layers.3";
        assert_eq!(
            role_of_name(&format!("{p}.linear_attn.in_proj_qkv.weight")),
            Some(Role::GdnQkv)
        );
        assert_eq!(
            role_of_name(&format!("{p}.linear_attn.A_log")),
            Some(Role::GdnALog)
        );
        assert_eq!(
            role_of_name(&format!("{p}.self_attn.o_proj.weight")),
            Some(Role::QsaO)
        );
        assert_eq!(
            role_of_name(&format!("{p}.self_attn.indexer.index_qk_proj.weight")),
            None
        );
        assert_eq!(
            role_of_name(&format!("{p}.mlp.shared_expert.down_proj.weight")),
            Some(Role::SharedDown)
        );
        assert_eq!(role_of_name(&format!("{p}.mlp.gate.weight")), None);
        assert_eq!(
            role_of_name("lm_head.weight"),
            SHARD_LM_HEAD.then_some(Role::LmHead)
        );
        assert_eq!(role_of_name("mtp.layers.0.self_attn.q_proj.weight"), None);
    }

    /// The MTP head stays replicated under the shard: no `mtp.*` tensor has a cut role, the
    /// census charges every one at its full bytes (whatever `parts`), the trunk entries next to
    /// them are cut, and the manifest of a head-keeping load declares the head's tensors.
    #[test]
    fn the_mtp_head_is_never_cut_and_the_census_charges_it_in_full() {
        use crate::expert_residency::{resident_split_ranks, resident_split_ranks_trunk};
        use hipfire_runtime::weight_manifest::WeightEntry;
        let flash = crate::config::compact_test_config();
        let manifest = Qwen4Manifest::build(&flash).unwrap();
        let head: Vec<WeightEntry> = manifest
            .weights
            .iter()
            .filter(|entry| entry.name.starts_with("mtp."))
            .cloned()
            .collect();
        // the head-keeping manifest declares the head's attention, shared expert and router
        for needle in [
            "self_attn.q_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.shared_expert.gate_proj.weight",
            "mlp.shared_expert.down_proj.weight",
        ] {
            assert!(
                head.iter().any(|entry| entry.name.contains(needle)),
                "mtp.* {needle} is in the manifest"
            );
        }
        // the head-dropping one declares none
        let headless = Qwen4Manifest::build(&flash.clone().into_headless().unwrap()).unwrap();
        assert!(headless
            .weights
            .iter()
            .all(|entry| !entry.name.starts_with("mtp.")));

        let dims = TrunkDims::from_config(&flash);
        let mode = QsaMode::GroupInterleaved;
        let bytes =
            |entry: &WeightEntry| Some(entry.logical_shape.iter().product::<usize>() as u64 * 2);
        for entry in &head {
            assert_eq!(role_of_name(&entry.name), None, "{}", entry.name);
            let full = bytes(entry).unwrap();
            for parts in [1, 2, 4] {
                assert_eq!(
                    trunk_share_bytes(&dims, mode, &entry.name, &entry.logical_shape, full, parts),
                    Ok(full),
                    "{} x{parts}",
                    entry.name
                );
            }
        }
        assert_eq!(
            resident_split_ranks_trunk(&head, bytes, 2, Some((&dims, mode))),
            resident_split_ranks(&head, bytes, 2)
        );
        // the trunk part of the same manifest IS cut: strictly fewer resident bytes per rank
        let (plain, _) = resident_split_ranks(&manifest.weights, bytes, 2).unwrap();
        let (sharded, _) =
            resident_split_ranks_trunk(&manifest.weights, bytes, 2, Some((&dims, mode))).unwrap();
        let (head_plain, _) = resident_split_ranks(&head, bytes, 2).unwrap();
        assert!(head_plain > 0 && sharded < plain);
        // ... and by exactly the trunk's cut: dropping the head from both leaves the same delta
        let trunk_only: Vec<WeightEntry> = manifest
            .weights
            .iter()
            .filter(|entry| !entry.name.starts_with("mtp."))
            .cloned()
            .collect();
        let (trunk_plain, _) = resident_split_ranks(&trunk_only, bytes, 2).unwrap();
        let (trunk_sharded, _) =
            resident_split_ranks_trunk(&trunk_only, bytes, 2, Some((&dims, mode))).unwrap();
        assert_eq!(plain - sharded, trunk_plain - trunk_sharded);
        assert_eq!(sharded - trunk_sharded, head_plain);
    }

    /// THE head policy: kept when the shard is requested AND the MTP switch is on (its default),
    /// dropped by `HIPFIRE_QWEN4_EP_TRUNK_SHARD_MTP=0`; no shard, no head policy.
    #[test]
    fn the_head_is_kept_under_the_shard_unless_the_mtp_switch_drops_it() {
        for requested in [false, true] {
            for env in [false, true] {
                assert_eq!(
                    keeps_mtp_given(requested, env),
                    requested && env,
                    "requested={requested} env={env}"
                );
                // an off shard drops nothing; an on shard drops unless the head is kept
                assert!(!drops_head_for(false, keeps_mtp_given(requested, env)));
                assert_eq!(
                    drops_head_for(true, keeps_mtp_given(requested, env)),
                    !(requested && env)
                );
            }
        }
        // this process requests no shard (no EP2 topology, no switch): no head policy at all
        assert!(!trunk_shard_keeps_mtp());
        assert!(!trunk_shard_drops_mtp_head());
        assert_eq!(TRUNK_SHARD_MTP_ENV, "HIPFIRE_QWEN4_EP_TRUNK_SHARD_MTP");
    }

    /// Census: the busiest rank keeps ceil(bytes * kept / total); replicated and unlisted
    /// tensors are unchanged; the Flash-Next MQ6 trunk lands near half.
    #[test]
    fn census_share_halves_the_cut_tensors() {
        let mq6 = |rows: usize, cols: usize| (rows * cols / 256 * 200) as u64;
        let name = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight";
        let full = mq6(10240, 2560);
        assert_eq!(
            trunk_share_bytes(
                &FN,
                QsaMode::GroupInterleaved,
                name,
                &[10240, 2560],
                full,
                2
            )
            .unwrap(),
            full / 2
        );
        let down = "model.language_model.layers.0.mlp.shared_expert.down_proj.weight";
        let bf16 = (2560 * 640 * 2) as u64;
        assert_eq!(
            trunk_share_bytes(&FN, QsaMode::GroupInterleaved, down, &[2560, 640], bf16, 2).unwrap(),
            bf16 * 384 / 640
        );
        let router = "model.language_model.layers.0.mlp.gate.weight";
        assert_eq!(
            trunk_share_bytes(&FN, QsaMode::GroupInterleaved, router, &[512, 2560], 99, 2).unwrap(),
            99
        );
        let k = "model.language_model.layers.3.self_attn.k_proj.weight";
        assert_eq!(
            trunk_share_bytes(&FN, QsaMode::GroupInterleaved, k, &[512, 2560], 1000, 2).unwrap(),
            1000
        );
        assert_eq!(
            trunk_share_bytes(
                &FN,
                QsaMode::GroupInterleaved,
                name,
                &[10240, 2560],
                full,
                1
            )
            .unwrap(),
            full
        );
    }

    /// The switch does nothing until the forward route is complete: a half-wired shard must
    /// never run, whatever the environment says.
    /// A quantized resident's device extent is rows x row_bytes, NOT `numel x dtype.size()`
    /// (card finding j6850: the MQ6 `[10240, 2560]` qkv is 20,480,000 B on device while
    /// `byte_size()` says 26,214,400).
    #[test]
    fn quantized_extent_is_the_geometry_not_byte_size() {
        let g = matrix_geometry(Dt::MQ6G256V2).unwrap();
        assert_eq!(row_bytes(g, 2560).unwrap() * 10240, 20_480_000);
        assert_ne!(10240 * 2560 * Dt::MQ6G256V2.size(), 20_480_000);
        let q8 = matrix_geometry(Dt::Q8_0).unwrap();
        assert_eq!(row_bytes(q8, 2560).unwrap() * 10240, 10240 * 2560 / 32 * 34);
    }

    #[test]
    fn switch_parse_and_default_context() {
        assert_eq!(parse_switch(None), ShardSwitch::Default);
        assert_eq!(parse_switch(Some("  ")), ShardSwitch::Default);
        for on in ["1", "true", "on", " 1 "] {
            assert_eq!(parse_switch(Some(on)), ShardSwitch::On, "{on}");
        }
        // the kill switch and anything unrecognised (including the retired lab value) is OFF
        for off in ["0", "off", "false", "lab", "yes", "2"] {
            assert_eq!(parse_switch(Some(off)), ShardSwitch::Off, "{off}");
        }
        // explicit values beat the context; the default needs two-rank EP over two gfx1201
        assert!(requested_for(ShardSwitch::On, false));
        assert!(!requested_for(ShardSwitch::Off, true));
        assert!(requested_for(ShardSwitch::Default, true));
        assert!(!requested_for(ShardSwitch::Default, false));
        assert!(default_applies_given(2, &["gfx1201", "gfx1201"]));
        assert!(!default_applies_given(1, &["gfx1201", "gfx1201"]));
        assert!(!default_applies_given(2, &["gfx1201"]));
        assert!(!default_applies_given(2, &["gfx1201", "gfx1100"]));
        assert!(!default_applies_given(4, &["gfx1201"; 4]));
    }

    #[test]
    fn dims_refuse_indivisible_geometry() {
        let mut d = FN;
        d.value_heads = 47;
        assert!(role_cut(&d, QsaMode::GroupInterleaved, Role::GdnZ, 0, 2).is_err());
        assert!(role_cut(&FN, QsaMode::GroupInterleaved, Role::GdnZ, 0, 1).is_err());
        assert!(role_cut(&FN, QsaMode::GroupInterleaved, Role::GdnZ, 2, 2).is_err());
        let mut d = FN;
        d.kv_heads = 3; // 24 q heads over 3 KV heads: group 8, odd parts
        assert!(role_cut(&d, QsaMode::KvSplit, Role::QsaK, 0, 2).is_err());
    }
}
