// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! qwen4_cpu_split_parity — lab gate for the D2 CPU-cold / GPU-hot routed-expert split
//! (expert hot set + `memory.offload_exec=cpu`, decode and verify windows of at most
//! four rows), on one real Flash-Next model.
//!
//! The model loads like the product loader's hot-set path (`qwen4_kld` loading,
//! plus the hot-set plan and tier owners): the
//! sidecar's ranking fills `--hot-budget-mib` of device experts, every other trunk expert
//! lives in the pinned host tier, and `attach_forward` arms the split per layer
//! (`Qwen4Bundle::split_layers`). Arming needs `HIPFIRE_OFFLOAD_EXEC=cpu`; the hot
//! set comes from an explicit sidecar path (`--hotset` or
//! `HIPFIRE_QWEN4_EXPERT_HOTSET`) on any arch (gfx1100 included), and gfx1201 also
//! takes the default sibling sidecar.
//!
//! `--untiered` is the shipped default shape instead: NO hot set, NO tier plan, the load's
//! placement unchanged (unset `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS` or `auto` resolves the
//! product loader's automatic placement through the one shared
//! `expert_residency::incumbent_auto_resident_layers`; a fixed `N` is only an optional
//! diagnostic override that host-maps trunk layers `N..`. The host-placed layers are READ
//! BACK from the placed weights and recorded with a placement digest and a
//! `placement_source`). The split is requested
//! by the lab force
//! `HIPFIRE_QWEN4_D2_DEFAULT=1` with `HIPFIRE_OFFLOAD_EXEC` UNSET (an explicit
//! `memory.offload_exec` always wins over the force, so any explicit value is refused here),
//! and every armed layer must be a whole-layer host owner: split owner kind `whole-layer`,
//! identity `cpu-whole-split-v1/<bitmap of the host-placed layers>`, placement all-cold.
//! `HIPFIRE_QWEN4_D2_RETURN=mapped|copy` (default `copy`; `mapped` on exact gfx1100) selects the cold-result
//! transport at load: `mapped` makes every all-cold layer's canonical combine read
//! the CPU-written mapped allocation instead of H2D-copied slots (a mixed-placement
//! layer keeps `copy`). Every layer reports its return mode, `output_copy_calls`,
//! `mapped_publications` and `mapped_retire_waits`; a gate checks that a mapped call
//! publishes once with zero cold-slot copies. The same-input gates run unchanged on
//! the tensor the combine actually read (the mapped view in mapped mode).
//! `HIPFIRE_QWEN4_D2_READY=sync|event` (default `sync`) selects the input-readiness
//! mechanism of `prepare` and `HIPFIRE_QWEN4_D2_JOIN=host|cpwait` (default `host`; `cpwait` on exact gfx1100, so a gfx1100 lab run needs `HIPFIRE_QWEN4_D2_JOIN=host`) the
//! join. Readiness is the configured `HIPFIRE_QWEN4_D2_READY` mode under either join:
//! the CP-wait join reports the same `sync`/`event` readiness as the host join. The
//! effective per-layer modes, the cumulative `ready_polls` (event queries),
//! `ready_fallback_syncs`, `cpwait_joins`, `cpwait_aborts` and `watchdog_fires`
//! counters are printed and written to the JSON (`d2_ready_modes`, `d2_join_modes`,
//! `d2_readiness_counters`); an `event` run gates on its layers having polled. The
//! lab parity runs drain the stream before the join, so a run in which any layer took
//! the CP-wait join is REFUSED (run it with `host`); the refusal of `cpwait` itself
//! (copy return, mixed layer, no stream-wait-value capability) is logged at load.
//! The all-CPU oracle reads the artifact's aggregate tensors at the D1 spans (tensor data
//! offset + expert * stride), re-derived here independently of the slab adapter and compared
//! with it, never through a bundle owner. The AllCold endpoint must reproduce that oracle
//! bitwise; there is no hot expert, so the natural run IS the all-cold placement and the
//! mixed-join gates are replaced by all-cold gates.
//! Real limits, not gates on a model or card: no arch, device or file name is consulted;
//! arming, owner kind and identity come from the bundle's own capability accessors. The
//! oracle reads the packed aggregate MQ4-XTS (QT44 gate/up + QT53 down) expert layout that
//! `artifact::hotset_aggregate_slabs` adapts and the CPU arithmetic kernels support; any
//! other expert format/layout is refused there (slab-arena, FP8, planar I4 have no host
//! aggregate to span), so the lab can only qualify what the shared slab adapter can span.
//!
//! Every comparison is SAME-INPUT: the split executor captures, inside the real forward
//! call of one layer, the exact rotated activation, the original route, the combine
//! target before the join, the expanded slot buffer the canonical combine consumed, the
//! combine target after it, and an all-GPU (pcie) re-run of the same routed tail on the
//! ORIGINAL ids into the same expanded buffer (`SplitParityRecord`). From those bytes
//! this tool computes, per layer call, with no second trajectory:
//!   * the all-CPU D1 arithmetic of every valid slot (`moe_cpu_host_slot_outputs`) from
//!     the model's ORIGINAL artifact bytes (the HFQ mmap through the tier plan's source
//!     spans: hot experts have no host copy in the tier owners) and a host canonical
//!     combine, which is first proven to reproduce the real GPU combine bitwise on the
//!     D2 expanded buffer ("twin" gate);
//!   * max abs / max rel (and relative L2) of the D2 joined target against the all-CPU
//!     target and against the all-GPU target, and the CPU-vs-GPU distance between the
//!     two reference routes for scale;
//!   * bitwise gates: hot slots of D2 vs the all-GPU slots, cold slots vs the all-CPU
//!     slots, rows whose valid slots are all hot vs the all-GPU target, rows whose valid
//!     slots are all cold vs the all-CPU target.
//!
//! Three AR runs (same teacher-forced or greedy token stream, same per-call inputs
//! within each call): `natural` (the real mixed placement), `all-hot` and `all-cold`
//! (placement endpoints forced inside the executor; every valid slot is owned by the GPU
//! resp. computed on the CPU from the original bytes). The endpoints must be bitwise
//! equal to the all-GPU resp. all-CPU targets of their own calls.
//!
//! Timing (Sol kill 3): the same token stream is run through the natural placement with
//! the executor's overlapped schedule and with the serial schedule (GPU tail drained
//! before the CPU slot arithmetic), identical kernels and bytes, ABBA ordered, and the
//! per-layer end-to-end wall (prepare entry to stream idle after the combine) is reported
//! (median / min / mean, steps `--warmup-steps` and later). Final logits of every step
//! must be bit-identical across all runs of the natural placement (full capture, overlapped
//! timing, serial timing).
//! Timing JSON retains every step hash and reports same-schedule repeats separately
//! from cross-schedule comparisons. Fewer than two runs per schedule leaves the
//! repeat gate `not_checked` (not a pass); admission requires `--timing-rounds 2`.
//!
//! Usage:
//!   HIPFIRE_OFFLOAD_EXEC=cpu \
//!   cargo run --release -p hipfire-arch-qwen4 --features lab \
//!     --example qwen4_cpu_split_parity -- MODEL.mq4 TEXT.txt --hot-budget-mib N \
//!     [--hotset SIDECAR] [--layers 4|0,1,2|all] [--offset 0] [--prompt-rows 32] \
//!     [--steps 16] [--rows 1] [--greedy] [--timing-rounds 2] [--warmup-steps 2] \
//!     [--no-endpoints] [--no-timing] [--json OUT.json]
//!   HIPFIRE_QWEN4_D2_DEFAULT=1 [HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=N optional diagnostic override]  (HIPFIRE_OFFLOAD_EXEC unset) \
//!   cargo run --release -p hipfire-arch-qwen4 --features lab \
//!     --example qwen4_cpu_split_parity -- MODEL.mq4 TEXT.txt --untiered \
//!     [--layers ...] [--steps 16] [--rows 1] [--json OUT.json]   (no --hot-budget-mib/--hotset)
//!
//! `--layers` selects the first N armed layers (a bare number), an explicit list, or all.
//! `--rows R` (1..=4) feeds R teacher-forced tokens per step through one narrow forward;
//! `--greedy` (rows 1 only) feeds back the argmax. `--prompt-rows` (>= 5) is prefilled on
//! the wide route first. Exit 0 only if every gate passes, 1 on a gate failure, 2 on an
//! error. Max abs / rel numbers are measurements, never gates (different arithmetic
//! routes); non-finite values fail.

use hipfire_arch_qwen4::admit_hfqm_artifact_on;
use hipfire_arch_qwen4::bundle::Qwen4Bundle;
use hipfire_arch_qwen4::expert_hotset as hotsets;
use hipfire_arch_qwen4::expert_residency::{host_placed_trunk_layers, placement_digest};
use hipfire_arch_qwen4::expert_slab_table::ExpertSlab;
use hipfire_arch_qwen4::expert_slabs::ExpertFormat;
use hipfire_arch_qwen4::expert_tier_plan::plan_expert_tiers_from_slabs;
use hipfire_dispatch::cpu_exec::{cpu_quant_for, moe_cpu_host_slot_outputs, CpuMoeExpert};
use hipfire_dispatch::cpu_split::{
    CpuShadow, D2ReturnMode, SplitEndpoint, SplitParityConfig, SplitParityMode, SplitParityRecord,
    SplitTiming,
};
use hipfire_runtime::device_mesh::DeviceMesh;
use hipfire_runtime::hfq::{HfqFile, HfqModelSource};
use hipfire_runtime::model_source::{ModelSource, SourcePayload};
use hipfire_runtime::ple_sidecar::{resolve_and_attach_ple_rows, ResolvePolicy};
use hipfire_runtime::tokenizer::Tokenizer;
use hipfire_runtime::weight_store::{fulfill_manifest_from_payloads, WeightOrigin};
use rdna_compute::{DType, Gpu};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::Arc;

type R<T> = Result<T, String>;

fn e<E: std::fmt::Display>(x: E) -> String {
    x.to_string()
}

const USAGE: &str = "usage: HIPFIRE_OFFLOAD_EXEC=cpu qwen4_cpu_split_parity MODEL.mq4 TEXT.txt \
    --hot-budget-mib N [--hotset SIDECAR] [--layers 4|0,1,2|all] [--offset 0] \
    [--prompt-rows 32] [--steps 16] [--rows 1] [--greedy] [--timing-rounds 2] \
    [--warmup-steps 2] [--no-endpoints] [--no-timing] [--json OUT.json]\n   or: \
    HIPFIRE_QWEN4_D2_DEFAULT=1 [HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=N optional diagnostic override; unset/auto = product placement] (HIPFIRE_OFFLOAD_EXEC unset) \
    qwen4_cpu_split_parity MODEL.mq4 TEXT.txt --untiered [same options, no --hot-budget-mib/--hotset]";

/// Rows the split serves at most (`CPU_SPLIT_MAX_ROWS`).
const MAX_SPLIT_ROWS: usize = 4;

// ── arguments ────────────────────────────────────────────────────────

enum LayerSel {
    First(usize),
    List(Vec<usize>),
    All,
}

struct Args {
    model: PathBuf,
    text: PathBuf,
    /// `Some` selects the tiered hot-set run; `None` is `--untiered`.
    hot_budget_mib: Option<u64>,
    hotset: Option<PathBuf>,
    layers: LayerSel,
    offset: usize,
    prompt_rows: usize,
    steps: usize,
    rows: usize,
    greedy: bool,
    timing_rounds: usize,
    warmup_steps: usize,
    endpoints: bool,
    timing: bool,
    json: Option<PathBuf>,
    untiered: bool,
}

fn parse_args() -> R<Args> {
    let mut positional = Vec::new();
    let mut hot_budget_mib: Option<u64> = None;
    let mut hotset = None;
    let mut layers = LayerSel::First(4);
    let mut offset = 0usize;
    let mut prompt_rows = 32usize;
    let mut steps = 16usize;
    let mut rows = 1usize;
    let mut greedy = false;
    let mut timing_rounds = 2usize;
    let mut warmup_steps = 2usize;
    let mut endpoints = true;
    let mut timing = true;
    let mut json = None;
    let mut untiered = false;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = |name: &str| {
            it.next()
                .ok_or_else(|| format!("{name} needs a value\n{USAGE}"))
        };
        let num = |name: &str, v: String| v.parse::<usize>().map_err(|x| format!("{name}: {x}"));
        match arg.as_str() {
            "--hot-budget-mib" => {
                hot_budget_mib = Some(
                    value("--hot-budget-mib")?
                        .parse()
                        .map_err(|x| format!("--hot-budget-mib: {x}"))?,
                )
            }
            "--hotset" => hotset = Some(PathBuf::from(value("--hotset")?)),
            "--untiered" => untiered = true,
            "--layers" => {
                let list = value("--layers")?;
                let list = list.trim();
                layers = if list == "all" {
                    LayerSel::All
                } else if !list.contains(',') {
                    LayerSel::First(num("--layers", list.to_string())?)
                } else {
                    LayerSel::List(
                        list.split(',')
                            .filter(|v| !v.trim().is_empty())
                            .map(|v| {
                                v.trim()
                                    .parse::<usize>()
                                    .map_err(|x| format!("--layers '{v}': {x}"))
                            })
                            .collect::<R<Vec<_>>>()?,
                    )
                };
            }
            "--offset" => offset = num("--offset", value("--offset")?)?,
            "--prompt-rows" => prompt_rows = num("--prompt-rows", value("--prompt-rows")?)?,
            "--steps" => steps = num("--steps", value("--steps")?)?,
            "--rows" => rows = num("--rows", value("--rows")?)?,
            "--greedy" => greedy = true,
            "--timing-rounds" => timing_rounds = num("--timing-rounds", value("--timing-rounds")?)?,
            "--warmup-steps" => warmup_steps = num("--warmup-steps", value("--warmup-steps")?)?,
            "--no-endpoints" => endpoints = false,
            "--no-timing" => timing = false,
            "--json" => json = Some(PathBuf::from(value("--json")?)),
            other if other.starts_with("--") => {
                return Err(format!("unknown argument {other}\n{USAGE}"))
            }
            other => positional.push(other.to_string()),
        }
    }
    let [model, text] =
        <[String; 2]>::try_from(positional).map_err(|_| format!("need MODEL and TEXT\n{USAGE}"))?;
    if untiered {
        if hot_budget_mib.is_some() || hotset.is_some() {
            return Err(
                "--untiered is the no-hot-set shape: drop --hot-budget-mib/--hotset".into(),
            );
        }
    } else {
        let mib = hot_budget_mib
            .ok_or_else(|| format!("--hot-budget-mib is required (or pass --untiered)\n{USAGE}"))?;
        if mib == 0 {
            return Err("--hot-budget-mib must be > 0".into());
        }
    }
    if rows == 0 || rows > MAX_SPLIT_ROWS {
        return Err(format!(
            "--rows {rows}: the split serves 1..={MAX_SPLIT_ROWS} rows"
        ));
    }
    if greedy && rows != 1 {
        return Err("--greedy feeds one argmax token per step: it needs --rows 1".into());
    }
    if prompt_rows <= MAX_SPLIT_ROWS {
        return Err(format!(
            "--prompt-rows {prompt_rows}: the prefill must be wider than {MAX_SPLIT_ROWS} rows so only decode/verify steps take the split"
        ));
    }
    if steps == 0 {
        return Err("--steps must be >= 1".into());
    }
    if timing && (timing_rounds == 0 || warmup_steps >= steps) {
        return Err(
            "--timing-rounds must be >= 1 and --warmup-steps < --steps (or pass --no-timing)"
                .into(),
        );
    }
    Ok(Args {
        model: PathBuf::from(model),
        text: PathBuf::from(text),
        hot_budget_mib,
        hotset,
        layers,
        offset,
        prompt_rows,
        steps,
        rows,
        greedy,
        timing_rounds,
        warmup_steps,
        endpoints,
        timing,
        json,
        untiered,
    })
}

// ── model ────────────────────────────────────────────────────────────

/// CPU-readable bytes of every expert of every trunk layer, served from the ORIGINAL
/// artifact mapping through the aggregate source spans (the bytes `materialize` copied
/// into the tier owners, and the bytes the D1/whole-layer host owners pin). Independent of
/// every owner and of host residency, so it resolves hot AND cold experts for the all-CPU
/// reference and the AllCold endpoint.
struct HfqShadow {
    /// Address and length of the artifact mapping covering every source span.
    base: usize,
    len: usize,
    /// `layer * experts + expert`.
    slabs: Vec<ExpertSlab>,
    experts: usize,
}

impl HfqShadow {
    fn bytes(&self, offset: usize, len: usize) -> Option<&[u8]> {
        let end = offset.checked_add(len)?;
        if end > self.len || len == 0 {
            return None;
        }
        // SAFETY: `base..base+len` is the read-only artifact mapping held by the model's
        // `HfqModelSource` (`Model::_source`, dropped after the bundle that holds this
        // shadow); the range was bounds-checked above and the mapping is never written.
        Some(unsafe { std::slice::from_raw_parts((self.base + offset) as *const u8, len) })
    }
}

impl CpuShadow for HfqShadow {
    fn expert(&self, layer: usize, expert: usize) -> Option<CpuMoeExpert<'_>> {
        if expert >= self.experts {
            return None;
        }
        let slab = self.slabs.get(layer * self.experts + expert)?;
        if !matches!(slab.format, ExpertFormat::Mq4Xts) {
            return None;
        }
        Some(CpuMoeExpert {
            gate_up_quant: cpu_quant_for(DType::MQ4G256V2)?,
            gate_up: self.bytes(slab.gate_up.offset, slab.gate_up.len)?,
            down_quant: cpu_quant_for(DType::MQ4G128V2)?,
            down: self.bytes(slab.down.offset, slab.down.len)?,
        })
    }
}

struct Model {
    gpu: Gpu,
    bundle: Qwen4Bundle,
    tokenizer: Tokenizer,
    vocab: usize,
    shadow: Arc<HfqShadow>,
    hot_experts: usize,
    /// Tier plan digest (tiered) or `untiered` (no plan exists).
    plan_digest: String,
    /// Diagnostic only: `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=N` when the run set a fixed count
    /// (trunk layers `0..N` resident); `None` when the product's automatic placement was
    /// resolved. The authoritative placement is `host_layers`.
    untiered_vram_layers: Option<usize>,
    /// Who produced the un-tiered placement: `env-fixed`, `product-resolver` or
    /// `product-resolver-resident`; `None` for a tiered run.
    placement_source: Option<&'static str>,
    /// Trunk layers with host-mapped routed experts in the placed weights (empty for a
    /// tiered run: the tier owners hold those experts).
    host_layers: Vec<usize>,
    /// Digest of `(trunk layer count, host_layers)`: equal across runs means equal placement.
    placement_digest: String,
    /// sha256 over every source span the oracle reads (receipt identity of the oracle).
    span_digest: String,
    /// Keeps the artifact mapping the shadow reads alive; dropped last.
    _source: HfqModelSource,
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The un-tiered lab shape: the split is requested only by the lab force, never by an
/// explicit `memory.offload_exec` (which always wins over the force and would change the
/// route being measured) and never against the `HIPFIRE_MOE_CPU_EXPERTS=0` kill-switch.
fn admit_untiered(
    explicit: Option<hipfire_config::memory::OffloadExec>,
    force: Option<bool>,
    kill_switch: bool,
) -> R<()> {
    if let Some(mode) = explicit {
        return Err(format!(
            "--untiered needs memory.offload_exec UNSET (HIPFIRE_OFFLOAD_EXEC unset): '{mode}' is explicit and wins over the D2 force"
        ));
    }
    if force != Some(true) {
        return Err(format!(
            "--untiered needs HIPFIRE_QWEN4_D2_DEFAULT=1 (developer.qwen4_d2_default is {force:?})"
        ));
    }
    if kill_switch {
        return Err("--untiered conflicts with HIPFIRE_MOE_CPU_EXPERTS=0".into());
    }
    Ok(())
}

/// Optional diagnostic fixed placement: `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=N` keeps trunk
/// layers `0..N` resident and host-maps the rest. Unset (or `auto`) returns `None`: the
/// product loader's automatic placement is then resolved through the shared
/// `expert_residency::incumbent_auto_resident_layers`. The placement the run actually has is
/// always read back from the placed weights, never from this value.
fn untiered_vram_layers(
    policy: Option<hipfire_arch_qwen4::expert_residency::ExpertVramLayers>,
    layers: usize,
) -> R<Option<usize>> {
    use hipfire_arch_qwen4::expert_residency::ExpertVramLayers;
    match policy {
        Some(ExpertVramLayers::Layers(n)) if n < layers => Ok(Some(n)),
        Some(ExpertVramLayers::Layers(n)) => Err(format!(
            "HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS={n} keeps all {layers} trunk layers resident: no host-mapped layer to arm"
        )),
        Some(ExpertVramLayers::Auto) | None => Ok(None),
    }
}

/// Re-derive every expert span from the artifact's aggregate tensors exactly as the D1 host
/// owner addresses them (tensor data offset + `expert * stride`, exact per-expert extent)
/// and require the slab table the oracle reads to equal it.
fn check_d1_spans(hfq: &HfqFile, layers: usize, experts: usize, slabs: &[ExpertSlab]) -> R<()> {
    let (gu_stride, dn_stride) = (
        ExpertFormat::Mq4Xts.gate_up_bytes(),
        ExpertFormat::Mq4Xts.down_bytes(),
    );
    if slabs.len() != layers * experts {
        return Err(format!(
            "slab table has {} entries, expected {layers} x {experts}",
            slabs.len()
        ));
    }
    for layer in 0..layers {
        let base = |suffix: &str| -> R<usize> {
            let name = format!("model.language_model.layers.{layer}.mlp{suffix}");
            Ok(hfq
                .find_tensor_info(&name)
                .ok_or_else(|| format!("aggregate tensor {name} is missing"))?
                .data_offset)
        };
        let (gu_base, dn_base) = (base(".experts.gate_up_proj")?, base(".experts.down_proj")?);
        for expert in 0..experts {
            let slab = &slabs[layer * experts + expert];
            let ok = matches!(slab.format, ExpertFormat::Mq4Xts)
                && slab.gate_up.offset == gu_base + expert * gu_stride
                && slab.gate_up.len == gu_stride
                && slab.down.offset == dn_base + expert * dn_stride
                && slab.down.len == dn_stride;
            if !ok {
                return Err(format!(
                    "layer {layer} expert {expert}: slab spans {slab:?} differ from the D1 aggregate spans (gate_up {}+{gu_stride}, down {}+{dn_stride})",
                    gu_base + expert * gu_stride,
                    dn_base + expert * dn_stride,
                ));
            }
        }
    }
    Ok(())
}

fn span_digest(slabs: &[ExpertSlab]) -> String {
    let mut digest = Sha256::new();
    for slab in slabs {
        for v in [
            slab.gate_up.offset,
            slab.gate_up.len,
            slab.down.offset,
            slab.down.len,
        ] {
            digest.update((v as u64).to_le_bytes());
        }
    }
    format!("{:x}", digest.finalize())
}

/// The `qwen4_kld` load plus (tiered) the product loader's hot-set path (sidecar bound to
/// the base, aggregate slabs, tier plan under `--hot-budget-mib`, trunk experts removed from
/// the manifest, tier owners materialized, lease handed to the bundle), or (`--untiered`)
/// the unchanged placement with no tier lease at all.
fn load(args: &Args, max_seq: usize) -> R<Model> {
    let path = args.model.as_path();
    let mut hfq = HfqFile::open(path).map_err(|x| format!("open HFQM {}: {x}", path.display()))?;
    let tokenizer =
        Tokenizer::from_hfq_metadata(&hfq.metadata_json).map_err(|x| format!("tokenizer: {x}"))?;
    let mut gpu = Gpu::init().map_err(e)?;
    if gpu.is_uma() {
        return Err(format!(
            "device {} is unified memory: the split needs a discrete GPU",
            gpu.arch
        ));
    }
    if args.untiered {
        admit_untiered(
            hipfire_config::memory::offload_exec_explicit(),
            hipfire_config::memory::qwen4_d2_default(),
            hipfire_config::developer_var("HIPFIRE_MOE_CPU_EXPERTS")
                .ok()
                .as_deref()
                == Some("0"),
        )?;
    } else if !hipfire_dispatch::cpu_exec::moe_cpu_experts_enabled()
        || hipfire_dispatch::cpu_exec::passback_enabled()
    {
        return Err(format!(
            "memory.offload_exec is '{}': the D2 split needs HIPFIRE_OFFLOAD_EXEC=cpu (not passback)",
            hipfire_config::memory::offload_exec()
        ));
    }
    let choice = resolve_and_attach_ple_rows(
        &mut hfq,
        path,
        hipfire_config::memory::ple_format(),
        &gpu.arch,
        ResolvePolicy {
            derive_missing_q4: false,
            progress: None,
        },
    )
    .map_err(|x| format!("qwen4 PLE rows: {x}"))?;
    eprintln!("qwen4_cpu_split_parity: ple rows: {}", choice.format.name());
    let receipt = admit_hfqm_artifact_on(&hfq, Some(&gpu.arch))
        .map_err(|x| format!("qwen4 admission: {x}"))?;
    let trunk_overlay =
        hipfire_arch_qwen4::resolve_trunk_overlay(&hfq, path, &gpu.arch, &receipt.config)
            .map_err(|x| format!("qwen4 trunk overlay: {x}"))?;
    let scratch_mode = hipfire_arch_qwen4::artifact::source_forward_scratch_mode(
        &gpu,
        &hfq,
        &receipt.config,
        &receipt.manifest,
    )
    .map_err(e)?;
    let npu_identity = hipfire_arch_qwen4::npu_trunk::model_index_identity(&hfq, &gpu.arch);
    let mesh = DeviceMesh::single().map_err(|x| format!("qwen4 mesh: {x}"))?;
    let expected = WeightOrigin::for_single(&mesh, &gpu);
    let (layers, experts) = (receipt.config.num_hidden_layers, receipt.config.num_experts);

    // ── oracle spans: the artifact's aggregate tensors at the D1 spans ──
    let slabs = hipfire_arch_qwen4::artifact::hotset_aggregate_slabs(
        &hfq,
        &receipt.config,
        &receipt.manifest,
    )?;
    check_d1_spans(&hfq, layers, experts, &slabs)?;
    let spans_digest = span_digest(&slabs);
    // The one mapping every source span lies in (also the shadow's backing store).
    let mut need = 0usize;
    for slab in &slabs {
        for span in [slab.gate_up, slab.down] {
            need = need.max(span.end().ok_or("source span end overflows")?);
        }
    }
    let (map_base, map_len) = {
        let all = hfq
            .data_range(0, need)
            .ok_or("the artifact mapping does not cover the source spans")?;
        (all.as_ptr() as usize, all.len())
    };
    let shadow = Arc::new(HfqShadow {
        base: map_base,
        len: map_len,
        slabs: slabs.clone(),
        experts,
    });
    // Self-check: the shadow serves exactly the artifact mapping's bytes.
    for (layer, expert) in [(0usize, 0usize), (layers - 1, experts - 1)] {
        let got = shadow
            .expert(layer, expert)
            .ok_or("shadow cannot serve a probe expert")?;
        let slab = &slabs[layer * experts + expert];
        let direct = hfq
            .data_range(slab.gate_up.offset, slab.gate_up.len)
            .ok_or("probe range")?;
        if got.gate_up != direct {
            return Err("shadow resolver disagrees with the artifact mapping".into());
        }
    }

    // Resolved once, before placement: the product's automatic placement is sized from them
    // and the bundle is assembled with the same values.
    let state_format = hipfire_arch_qwen4::resolve_state_format(
        &hipfire_runtime::config::get().kv_mode,
        &std::env::var("HIPFIRE_STATE_QUANT").unwrap_or_default(),
        &gpu,
        &receipt.config,
    )?;
    let backend = hipfire_arch_qwen4::Qwen4KvBackend::automatic(&gpu);
    let mut manifest = receipt.manifest;
    let mut placements = receipt.placements;
    let (tier_lease, hot_experts, plan_digest, untiered_layers, placement_source) = if let Some(
        mib,
    ) =
        args.hot_budget_mib
    {
        // ── hot set: policy, sidecar, plan ───────────────────────────
        let policy = match &args.hotset {
            Some(p) => hotsets::HotsetPolicy::Explicit(p.clone()),
            None => hotsets::HotsetPolicy::from_env()?,
        };
        let env = hotsets::HotsetEnv {
            gpu_arch: &gpu.arch,
            unified_memory: false,
            expert_parallel: false,
            explicit_vram_layers: false,
            source: match &manifest.identity.recipe {
                hipfire_arch_qwen4::weights::Qwen4TierRecipe::FlashNextPacked { .. } => {
                    hotsets::HotsetSource::Aggregate
                }
                hipfire_arch_qwen4::weights::Qwen4TierRecipe::FlashNextSlab { .. } => {
                    hotsets::HotsetSource::SlabArena
                }
                hipfire_arch_qwen4::weights::Qwen4TierRecipe::StockFp8 => {
                    hotsets::HotsetSource::StockFp8
                }
                _ => hotsets::HotsetSource::Nonaggregate("not an aggregate MQ4 artifact".into()),
            },
        };
        let sidecar_path = hotsets::preflight(&policy, path, &env).map_err(|skip| {
            skip.log_line(&policy.sidecar_path(path))
                .unwrap_or_else(|| skip.to_string())
        })?;
        let base = hotsets::HotsetBase::from_hfq(&hfq)?;
        let sidecar = hotsets::HotsetSidecar::load_bound(
            &sidecar_path,
            receipt.config.num_hidden_layers,
            receipt.config.num_experts,
            receipt.config.num_experts_per_tok,
            &base,
        )
        .map_err(|skip| skip.to_string())?;
        manifest.check_expert_tiers().map_err(e)?;
        let budget = mib << 20;
        let plan = plan_expert_tiers_from_slabs(&slabs, sidecar.rank(), budget).map_err(e)?;
        hotsets::require_experts(plan.device_experts()).map_err(|skip| skip.to_string())?;
        plan.check_executable(&gpu.arch).map_err(e)?;
        let hot_experts = plan.device_experts();
        let plan = Arc::new(plan);
        let plan_digest = hex(&plan.logical_digest());

        // ── manifest: trunk experts are tier-owned, the MTP layer's stay host-mapped ──
        let mtp_moved = hipfire_arch_qwen4::expert_residency::place_routed_experts(
            &mut manifest.weights,
            layers,
        );
        manifest = manifest.with_expert_tiers().map_err(e)?;
        {
            let keep: BTreeSet<_> = manifest.resident_keys().collect();
            placements
                .retain(|placement| keep.contains(&(placement.name.as_str(), placement.layer)));
        }
        eprintln!(
            "qwen4_cpu_split_parity: hot set {} sha256 {}, budget {mib} MiB, {hot_experts} experts in VRAM, plan digest {}, {mtp_moved} MTP expert tensors host-mapped",
            sidecar_path.display(),
            sidecar.sha256(),
            &plan_digest[..16],
        );
        let lease = hipfire_arch_qwen4::expert_tier_owners::materialize(
            &mut gpu,
            plan.clone(),
            &|offset, len| {
                let offset = usize::try_from(offset)
                    .map_err(|_| "hotset source offset overflow".to_string())?;
                let len = usize::try_from(len)
                    .map_err(|_| "hotset source length overflow".to_string())?;
                hfq.data_range(offset, len).ok_or_else(|| {
                    format!("source span {offset}+{len} is outside the artifact mapping")
                })
            },
        )
        .map_err(|x| format!("expert tier materialize: {x}"))?;
        lease
            .bind_hotset_sidecar_sha256(sidecar.sha256())
            .map_err(|x| format!("hotset numerical identity: {x}"))?;
        (Some(lease), hot_experts, plan_digest, None, None)
    } else {
        // ── un-tiered: no hot set, no tier lease. Unset (or `auto`) resolves the product
        // loader's automatic placement through the one shared resolver; a fixed N is an
        // optional diagnostic override ──
        use hipfire_arch_qwen4::expert_residency as residency;
        let vram_layers = untiered_vram_layers(residency::expert_vram_layers_from_env()?, layers)?;
        let placement_source = match vram_layers {
            Some(n) => {
                let moved = residency::place_routed_experts(&mut manifest.weights, n);
                eprintln!(
                    "qwen4_cpu_split_parity: un-tiered placement (diagnostic N={n}): routed experts of trunk layers 0..{n} in VRAM, {moved} expert tensors host-mapped"
                );
                "env-fixed"
            }
            None => {
                let recipe = &manifest.identity.recipe;
                let slab_recipe = matches!(
                    recipe,
                    hipfire_arch_qwen4::weights::Qwen4TierRecipe::FlashNextSlab { .. }
                );
                // Entry sizes as the product loader sizes them: a slab layer arena from the
                // recipe's plan, anything else from the mapped tensor.
                let bytes_of =
                    |entry: &hipfire_runtime::weight_manifest::WeightEntry| -> Option<u64> {
                        if let hipfire_arch_qwen4::weights::Qwen4TierRecipe::FlashNextSlab {
                            arenas,
                            ..
                        } = recipe
                        {
                            if entry
                                .name
                                .ends_with(hipfire_arch_qwen4::expert_arena::ARENA_SUFFIX)
                            {
                                return entry
                                    .layer
                                    .and_then(|layer| arenas.layer(layer))
                                    .map(|layer| layer.bytes as u64);
                            }
                        }
                        hfq.tensor_data(&entry.name)
                            .map(|(_, bytes)| bytes.len() as u64)
                    };
                let vmm_granularity = match backend {
                    hipfire_arch_qwen4::Qwen4KvBackend::Vmm => gpu
                        .vmm_recommended_granularity()
                        .map_err(|x| format!("qwen4: VMM granularity: {x}"))?,
                    hipfire_arch_qwen4::Qwen4KvBackend::Legacy => 1,
                };
                let resolved = residency::incumbent_auto_resident_layers(
                    &gpu,
                    &receipt.config,
                    &manifest.weights,
                    &bytes_of,
                    backend,
                    max_seq,
                    vmm_granularity,
                    state_format,
                    scratch_mode,
                    residency::dedicated_slab_stage(
                        residency::expert_stage_placement_enabled(&gpu.arch),
                        slab_recipe,
                        false,
                    ),
                    0,
                )?;
                match resolved {
                    Some(n) => {
                        let moved = residency::place_routed_experts(&mut manifest.weights, n);
                        eprintln!(
                            "qwen4_cpu_split_parity: routed experts: product automatic placement (expert_residency::incumbent_auto_resident_layers): layers 0..{n} in VRAM, {moved} tensors host-mapped"
                        );
                        "product-resolver"
                    }
                    None => {
                        eprintln!(
                            "qwen4_cpu_split_parity: routed experts: product automatic placement keeps every expert resident"
                        );
                        "product-resolver-resident"
                    }
                }
            }
        };
        (
            None,
            0,
            "untiered".to_string(),
            vram_layers,
            Some(placement_source),
        )
    };
    let host_layers = host_placed_trunk_layers(&manifest.weights);
    let placement_digest = placement_digest(layers, &host_layers);
    if args.untiered {
        if host_layers.is_empty() {
            return Err(
                "the resolved placement has no host-mapped trunk layer (the product's automatic placement keeps every expert resident on this card): nothing for the whole-layer split to own; set a diagnostic HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=N so the card hosts experts"
                    .into(),
            );
        }
        eprintln!(
            "qwen4_cpu_split_parity: resolved placement: {} of {layers} trunk layers host-placed {host_layers:?}, placement digest {placement_digest}, no hot set, no tier lease; D1 span digest {}",
            host_layers.len(),
            &spans_digest[..16]
        );
    }

    let source = HfqModelSource::from_hfq(hfq);
    let transaction = fulfill_manifest_from_payloads(
        &manifest.weights,
        &mesh,
        layers,
        &mut gpu,
        expected,
        |entry| {
            if let Some(payload) = hipfire_arch_qwen4::expert_arena::recipe_arena_payload(
                &source,
                &manifest.identity.recipe,
                entry,
            ) {
                return payload;
            }
            if entry.residency == hipfire_runtime::weight_manifest::WeightResidency::HostMapped {
                let (info, bytes) = source
                    .tensor_data(&entry.name)
                    .ok_or_else(|| format!("missing resident tensor '{}'", entry.name))?;
                return Ok(SourcePayload::Borrowed { info, bytes });
            }
            source
                .tensor_range(&entry.name)
                .map_err(|x| x.to_string())?
                .map(SourcePayload::Range)
                .ok_or_else(|| format!("missing tensor '{}'", entry.name))
        },
    )
    .map_err(|x| format!("qwen4 manifest fulfillment: {x}"))?;
    let vocab = receipt.config.vocab_size;
    let mut bundle = Qwen4Bundle::assemble_with_ple_rows(
        receipt.config,
        &manifest.identity.recipe,
        transaction,
        &placements,
        &mut gpu,
        max_seq,
        receipt.ple,
        receipt.ple_rows,
        state_format,
        backend,
        tier_lease,
    )
    .map_err(|x| format!("qwen4 bundle assembly: {x}"))?;
    bundle.attach_npu_trunk_identity(npu_identity);
    if let Some(overlay) = trunk_overlay {
        bundle
            .attach_trunk_overlay(overlay)
            .map_err(|x| format!("qwen4 trunk overlay: {x}"))?;
    }
    bundle
        .attach_forward(&mut gpu, max_seq, scratch_mode)
        .map_err(|x| format!("qwen4 forward setup: {x}"))?;
    eprintln!(
        "qwen4_cpu_split_parity: loaded {} on {} (vocab {vocab})",
        path.display(),
        gpu.arch
    );
    let resolved_format = bundle.state_format();
    eprintln!(
        "[qwen4_cpu_split_parity] resolved kv_mode={:?} qsa_state={:?} gdn_state={:?}",
        hipfire_runtime::config::get().kv_mode,
        resolved_format.qsa,
        resolved_format.gdn
    );
    Ok(Model {
        gpu,
        bundle,
        tokenizer,
        vocab,
        shadow,
        hot_experts,
        plan_digest,
        untiered_vram_layers: untiered_layers,
        host_layers,
        placement_digest,
        placement_source,
        span_digest: spans_digest,
        _source: source,
    })
}

// ── arithmetic helpers ───────────────────────────────────────────────

fn valid_id(id: i32, n_experts: usize) -> bool {
    id >= 0 && (id as usize) < n_experts
}

/// The shared host canonical combine of the BF16 decode route
/// (`cpu_exec::moe_canonical_combine_reference`): each row starts at `rt(target)` and adds
/// its valid slots in ASCENDING (expert id, rank), `acc = rt(acc + rt(rt(down) * rt(w)))`.
/// Proven against the real GPU combine on every D2 call (the "twin" gate) before it is
/// applied to the reference expanded buffers.
fn combine(
    dim: usize,
    rows: usize,
    top_k: usize,
    n_experts: usize,
    ids: &[i32],
    weights: &[f32],
    expanded: &[f32],
    target_in: &[f32],
) -> Vec<f32> {
    let mut out = target_in[..rows * dim].to_vec();
    hipfire_dispatch::cpu_exec::moe_canonical_combine_reference(
        dim, rows, top_k, n_experts, ids, weights, expanded, &mut out,
    );
    out
}

fn bit_mismatch(a: &[f32], b: &[f32]) -> u64 {
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count() as u64
}

/// Accumulated distance of `a` to a reference `r`.
#[derive(Clone, Copy, Default)]
struct Diff {
    max_abs: f64,
    /// `max |a-r| / max(|a|, |r|, 1e-6)`.
    max_rel: f64,
    ref_max: f64,
    sum_d2: f64,
    sum_r2: f64,
}

impl Diff {
    fn add(&mut self, a: &[f32], r: &[f32]) {
        for (&x, &y) in a.iter().zip(r) {
            let (x, y) = (x as f64, y as f64);
            let d = (x - y).abs();
            let rel = d / x.abs().max(y.abs()).max(1e-6);
            self.max_abs = self.max_abs.max(d);
            self.max_rel = self.max_rel.max(rel);
            self.ref_max = self.ref_max.max(y.abs());
            self.sum_d2 += d * d;
            self.sum_r2 += y * y;
        }
    }

    fn merge(&mut self, o: &Diff) {
        self.max_abs = self.max_abs.max(o.max_abs);
        self.max_rel = self.max_rel.max(o.max_rel);
        self.ref_max = self.ref_max.max(o.ref_max);
        self.sum_d2 += o.sum_d2;
        self.sum_r2 += o.sum_r2;
    }

    fn rel_l2(&self) -> f64 {
        if self.sum_r2 > 0.0 {
            (self.sum_d2 / self.sum_r2).sqrt()
        } else {
            0.0
        }
    }

    fn scaled_max_abs(&self) -> f64 {
        if self.ref_max > 0.0 {
            self.max_abs / self.ref_max
        } else {
            0.0
        }
    }

    fn json(&self) -> Value {
        json!({"max_abs": self.max_abs, "max_rel": self.max_rel, "rel_l2": self.rel_l2(), "scaled_max_abs": self.scaled_max_abs()})
    }

    fn short(&self) -> String {
        format!("{:.3e}/{:.3e}", self.max_abs, self.max_rel)
    }
}

// ── per-call analysis ────────────────────────────────────────────────

#[derive(Default)]
struct LayerAgg {
    calls: u64,
    rows: u64,
    hot_slots: u64,
    cold_slots: u64,
    invalid_slots: u64,
    rows_hot: u64,
    rows_cold: u64,
    rows_mixed: u64,
    rows_empty: u64,
    /// Elements whose bits differ.
    twin_mismatch: u64,
    hot_slot_mismatch: u64,
    cold_slot_mismatch: u64,
    hot_row_mismatch: u64,
    cold_row_mismatch: u64,
    nonfinite: u64,
    /// Experts the record's placement mask puts on the device (0 for a whole-layer owner).
    placement_hot: u64,
    /// Calls that read the mapped cold-result view / the original H2D copy transport.
    mapped_calls: u64,
    copy_calls: u64,
    /// Cold-slot H2D enqueues, mapped generations published and retirements that
    /// actually blocked, summed over this layer's recorded calls.
    output_copy_calls: u64,
    mapped_publications: u64,
    mapped_retire_waits: u64,
    /// Calls whose counters contradict their return mode (mapped: exactly one
    /// publication and zero output copies; copy: no publication).
    transport_violations: u64,
    d2_vs_cpu: Diff,
    d2_vs_gpu: Diff,
    cpu_vs_gpu: Diff,
    /// Slot level, raw down values: cold slots of D2 vs the GPU arithmetic and hot slots
    /// vs the CPU arithmetic (the distance between the two arithmetic routes).
    cold_slots_vs_gpu: Diff,
    hot_slots_vs_cpu: Diff,
}

fn finite(v: &[f32]) -> u64 {
    v.iter().filter(|x| !x.is_finite()).count() as u64
}

fn analyze(rec: &SplitParityRecord, shadow: &HfqShadow, agg: &mut LayerAgg) -> R<()> {
    let (rows, dim, top_k) = (rec.rows, rec.dim, rec.top_k);
    let n_experts = rec.placement.len();
    let slots = rows * top_k;
    if rows == 0
        || rows > MAX_SPLIT_ROWS
        || rec.x_rot.len() != rows * dim
        || rec.ids.len() != slots
        || rec.weights.len() != slots
        || rec.target_in.len() != rows * dim
        || rec.d2_target.len() != rows * dim
        || rec.d2_expanded.len() != slots * dim
        || rec.gpu_expanded.len() != slots * dim
    {
        return Err(format!(
            "layer {}: parity record has inconsistent extents",
            rec.layer
        ));
    }

    // All-CPU D1 arithmetic of every valid slot from the original bytes.
    let host_slots: Vec<u32> = (0..slots)
        .filter(|&s| valid_id(rec.ids[s], n_experts))
        .map(|s| s as u32)
        .collect();
    let resolved: Vec<CpuMoeExpert<'_>> = host_slots
        .iter()
        .map(|&s| {
            shadow
                .expert(rec.layer, rec.ids[s as usize] as usize)
                .ok_or_else(|| {
                    format!(
                        "layer {}: no original bytes for expert {}",
                        rec.layer, rec.ids[s as usize]
                    )
                })
        })
        .collect::<R<_>>()?;
    let mut cpu_expanded = vec![f32::NAN; slots * dim];
    moe_cpu_host_slot_outputs(
        dim,
        rec.mi,
        rows,
        top_k,
        n_experts,
        &rec.ids,
        None,
        &host_slots,
        &resolved,
        &rec.x_rot,
        false,
        &mut cpu_expanded,
    )
    .map_err(|x| format!("layer {}: all-CPU reference: {x:?}", rec.layer))?;

    let target = |expanded: &[f32]| {
        combine(
            dim,
            rows,
            top_k,
            n_experts,
            &rec.ids,
            &rec.weights,
            expanded,
            &rec.target_in,
        )
    };
    let twin = target(&rec.d2_expanded);
    let t_cpu = target(&cpu_expanded);
    let t_gpu = target(&rec.gpu_expanded);

    agg.calls += 1;
    let transport = rec.transport;
    match transport.mode {
        D2ReturnMode::Mapped => {
            agg.mapped_calls += 1;
            if transport.output_copy_calls != 0 || transport.mapped_publications != 1 {
                agg.transport_violations += 1;
            }
        }
        D2ReturnMode::Copy => {
            agg.copy_calls += 1;
            if transport.mapped_publications != 0 {
                agg.transport_violations += 1;
            }
        }
    }
    agg.output_copy_calls += transport.output_copy_calls;
    agg.mapped_publications += transport.mapped_publications;
    agg.mapped_retire_waits += transport.mapped_retire_waits;
    agg.placement_hot += rec.placement.iter().filter(|&&p| p == 0).count() as u64;
    agg.rows += rows as u64;
    agg.twin_mismatch += bit_mismatch(&twin, &rec.d2_target);
    agg.d2_vs_cpu.add(&rec.d2_target, &t_cpu);
    agg.d2_vs_gpu.add(&rec.d2_target, &t_gpu);
    agg.cpu_vs_gpu.add(&t_cpu, &t_gpu);

    // Effective owner of every slot (endpoints override the placement).
    let owner = |s: usize| -> Option<u8> {
        let id = rec.ids[s];
        valid_id(id, n_experts).then(|| match rec.endpoint {
            SplitEndpoint::Natural => rec.placement[id as usize],
            SplitEndpoint::AllHot => 0,
            SplitEndpoint::AllCold => 1,
        })
    };
    let mut nonfinite = finite(&rec.d2_target) + finite(&t_cpu) + finite(&t_gpu);
    for s in 0..slots {
        let range = s * dim..(s + 1) * dim;
        match owner(s) {
            None => agg.invalid_slots += 1,
            Some(0) => {
                agg.hot_slots += 1;
                agg.hot_slot_mismatch += bit_mismatch(
                    &rec.d2_expanded[range.clone()],
                    &rec.gpu_expanded[range.clone()],
                );
                agg.hot_slots_vs_cpu.add(
                    &rec.d2_expanded[range.clone()],
                    &cpu_expanded[range.clone()],
                );
                nonfinite += finite(&rec.d2_expanded[range.clone()])
                    + finite(&rec.gpu_expanded[range.clone()])
                    + finite(&cpu_expanded[range]);
            }
            Some(_) => {
                agg.cold_slots += 1;
                agg.cold_slot_mismatch += bit_mismatch(
                    &rec.d2_expanded[range.clone()],
                    &cpu_expanded[range.clone()],
                );
                agg.cold_slots_vs_gpu.add(
                    &rec.d2_expanded[range.clone()],
                    &rec.gpu_expanded[range.clone()],
                );
                nonfinite += finite(&rec.d2_expanded[range.clone()])
                    + finite(&rec.gpu_expanded[range.clone()])
                    + finite(&cpu_expanded[range]);
            }
        }
    }
    agg.nonfinite += nonfinite;

    // Rows whose valid slots all sit on one side: the join degenerates to that side's
    // route, so the joined row must equal it bitwise.
    for t in 0..rows {
        let owners: Vec<u8> = (0..top_k).filter_map(|j| owner(t * top_k + j)).collect();
        let row = t * dim..(t + 1) * dim;
        if owners.is_empty() {
            agg.rows_empty += 1;
        } else if owners.iter().all(|&o| o == 0) {
            agg.rows_hot += 1;
            agg.hot_row_mismatch += bit_mismatch(&rec.d2_target[row.clone()], &t_gpu[row]);
        } else if owners.iter().all(|&o| o == 1) {
            agg.rows_cold += 1;
            agg.cold_row_mismatch += bit_mismatch(&rec.d2_target[row.clone()], &t_cpu[row]);
        } else {
            agg.rows_mixed += 1;
        }
    }
    Ok(())
}

// ── runs ─────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
enum Schedule {
    Overlapped,
    Serial,
}

struct PassSpec {
    name: String,
    mode: SplitParityMode,
    endpoint: SplitEndpoint,
    schedule: Schedule,
}

struct PassOut {
    /// sha256 of every step's logits (all rows), in order.
    digests: Vec<String>,
}

/// Zero-based first divergent step, including a missing step in either run.
fn first_digest_mismatch(a: &[String], b: &[String]) -> Option<usize> {
    a.iter()
        .zip(b)
        .position(|(x, y)| x != y)
        .or_else(|| (a.len() != b.len()).then_some(a.len().min(b.len())))
}

struct TimingRun {
    name: String,
    round: usize,
    schedule: Schedule,
    digests: Vec<String>,
}

/// Same-schedule pairs are repeats; different schedules are a separate diagnostic.
fn timing_comparisons(runs: &[TimingRun], same_schedule: bool) -> (Vec<Value>, bool) {
    let mut comparisons = Vec::new();
    let mut all_match = true;
    for (i, a) in runs.iter().enumerate() {
        for b in &runs[i + 1..] {
            if (a.schedule == b.schedule) != same_schedule {
                continue;
            }
            let first = first_digest_mismatch(&a.digests, &b.digests);
            all_match &= first.is_none();
            comparisons.push(json!({
                "a": a.name, "b": b.name, "matching": first.is_none(),
                "first_mismatch_step": first,
            }));
        }
    }
    (comparisons, all_match)
}

fn argmax(row: &[f32]) -> u32 {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// One AR pass: reset, prefill `prompt_rows` on the wide route, then `steps` narrow
/// forwards of `rows` tokens, draining the parity records after every step. `sink`
/// receives `(step, records)`.
fn run_pass(
    m: &mut Model,
    args: &Args,
    spec: &PassSpec,
    tokens: &[u32],
    layers: &[usize],
    mut sink: impl FnMut(usize, Vec<SplitParityRecord>) -> R<()>,
) -> R<PassOut> {
    let shadow: Arc<dyn CpuShadow> = m.shadow.clone();
    m.bundle
        .split_parity_configure(SplitParityConfig {
            mode: spec.mode,
            layers: Some(layers.to_vec()),
            endpoint: spec.endpoint,
            serial: spec.schedule == Schedule::Serial,
            shadow: Some(shadow),
            max_records: hipfire_dispatch::cpu_split::SPLIT_PARITY_MAX_RECORDS,
        })
        .map_err(|x| format!("{}: configure parity: {x}", spec.name))?;
    m.bundle.reset(&mut m.gpu).map_err(e)?;
    let vocab = m.vocab;
    let all = m.gpu.zeros(&[args.rows * vocab], DType::F32).map_err(e)?;
    let row1 = all.sub_offset(0, vocab);
    let view = all.sub_offset(0, args.rows * vocab);

    m.bundle
        .forward_chunk_final(&mut m.gpu, &tokens[..args.prompt_rows], &row1, None)
        .map_err(|x| format!("{}: prefill: {x}", spec.name))?;
    let stray = m.bundle.split_take_parity_records();
    if !stray.is_empty() {
        return Err(format!(
            "{}: the split ran inside the {}-row prefill ({} records)",
            spec.name,
            args.prompt_rows,
            stray.len()
        ));
    }
    let mut next_token = if args.greedy {
        let first = m.gpu.download_f32(&row1).map_err(e)?;
        Some(argmax(&first))
    } else {
        None
    };
    let mut digests = Vec::with_capacity(args.steps);
    for step in 0..args.steps {
        let at = args.prompt_rows + step * args.rows;
        let logits_host = if args.rows == 1 {
            let token = next_token.unwrap_or(tokens[at]);
            m.bundle
                .forward_token(&mut m.gpu, token, &row1, None)
                .map_err(|x| format!("{}: step {step}: {x}", spec.name))?;
            m.gpu.download_f32(&row1).map_err(e)?
        } else {
            m.bundle
                .forward_chunk(&mut m.gpu, &tokens[at..at + args.rows], &view, None)
                .map_err(|x| format!("{}: step {step}: {x}", spec.name))?;
            m.gpu.download_f32(&view).map_err(e)?
        };
        if logits_host.iter().any(|v| !v.is_finite()) {
            return Err(format!("{}: step {step}: non-finite logits", spec.name));
        }
        if args.greedy {
            next_token = Some(argmax(&logits_host));
        }
        let mut digest = Sha256::new();
        for v in &logits_host {
            digest.update(v.to_bits().to_le_bytes());
        }
        digests.push(format!("{:x}", digest.finalize()));
        let recs = m.bundle.split_take_parity_records();
        if recs.len() != layers.len()
            || recs.iter().map(|r| r.layer).collect::<BTreeSet<_>>()
                != layers.iter().copied().collect()
        {
            return Err(format!(
                "{}: step {step}: expected one parity record per selected layer ({} layers), got layers {:?}",
                spec.name,
                layers.len(),
                recs.iter().map(|r| r.layer).collect::<Vec<_>>()
            ));
        }
        if let Some(bad) = recs
            .iter()
            .find(|r| r.rows != args.rows || r.endpoint != spec.endpoint)
        {
            return Err(format!(
                "{}: step {step}: layer {} record has rows {} endpoint {:?}, expected rows {} endpoint {:?}",
                spec.name, bad.layer, bad.rows, bad.endpoint, args.rows, spec.endpoint
            ));
        }
        sink(step, recs)?;
    }
    m.gpu.hip.device_synchronize().map_err(e)?;
    m.bundle
        .split_parity_configure(SplitParityConfig {
            mode: SplitParityMode::Off,
            layers: None,
            endpoint: SplitEndpoint::Natural,
            serial: false,
            shadow: None,
            max_records: 0,
        })
        .map_err(|x| format!("{}: clear parity: {x}", spec.name))?;
    Ok(PassOut { digests })
}

// ── timing ───────────────────────────────────────────────────────────

fn median(v: &mut Vec<u64>) -> u64 {
    if v.is_empty() {
        return 0;
    }
    v.sort_unstable();
    v[v.len() / 2]
}

#[derive(Default)]
struct TimingAgg {
    wall: Vec<u64>,
    prepare: Vec<u64>,
    enqueue: Vec<u64>,
    cpu: Vec<u64>,
    join: Vec<u64>,
    tail: Vec<u64>,
    cold: u64,
    hot: u64,
    n: u64,
}

impl TimingAgg {
    fn add(&mut self, t: &SplitTiming) {
        self.wall.push(t.wall_ns);
        self.prepare.push(t.prepare_ns);
        self.enqueue.push(t.gpu_enqueue_ns);
        self.cpu.push(t.join_cpu_ns);
        self.join.push(t.join_total_ns);
        self.tail.push(t.tail_ns);
        self.cold += t.cold_slots as u64;
        self.hot += t.hot_slots as u64;
        self.n += 1;
    }

    fn stats(&self) -> Value {
        let mut w = self.wall.clone();
        let mean = if w.is_empty() {
            0.0
        } else {
            w.iter().sum::<u64>() as f64 / w.len() as f64
        };
        let min = w.iter().copied().min().unwrap_or(0);
        json!({
            "calls": self.n,
            "wall_median_ns": median(&mut w),
            "wall_min_ns": min,
            "wall_mean_ns": mean,
            "prepare_median_ns": median(&mut self.prepare.clone()),
            "gpu_enqueue_median_ns": median(&mut self.enqueue.clone()),
            "join_cpu_median_ns": median(&mut self.cpu.clone()),
            "join_total_median_ns": median(&mut self.join.clone()),
            "tail_median_ns": median(&mut self.tail.clone()),
            "hot_slots_per_call": if self.n == 0 { 0.0 } else { self.hot as f64 / self.n as f64 },
            "cold_slots_per_call": if self.n == 0 { 0.0 } else { self.cold as f64 / self.n as f64 },
        })
    }

    fn wall_median(&self) -> u64 {
        median(&mut self.wall.clone())
    }
}

// ── gates ────────────────────────────────────────────────────────────

struct Gate {
    name: String,
    ok: bool,
    detail: String,
}

fn gate(gates: &mut Vec<Gate>, name: impl Into<String>, ok: bool, detail: impl Into<String>) {
    let g = Gate {
        name: name.into(),
        ok,
        detail: detail.into(),
    };
    println!(
        "  GATE {} {}: {}",
        if g.ok { "PASS" } else { "FAIL" },
        g.name,
        g.detail
    );
    gates.push(g);
}

fn print_layers(aggs: &BTreeMap<usize, LayerAgg>) {
    println!(
        "  {:>5} {:>5} {:>9} {:>14} {:>23} {:>23} {:>23} {:>23} {:>23}",
        "layer",
        "calls",
        "hot/cold",
        "rows h/c/m",
        "D2~CPU abs/rel",
        "D2~GPU abs/rel",
        "CPU~GPU abs/rel",
        "coldslot~GPU",
        "hotslot~CPU"
    );
    for (layer, a) in aggs {
        println!(
            "  {:>5} {:>5} {:>9} {:>14} {:>23} {:>23} {:>23} {:>23} {:>23}",
            layer,
            a.calls,
            format!("{}/{}", a.hot_slots, a.cold_slots),
            format!("{}/{}/{}", a.rows_hot, a.rows_cold, a.rows_mixed),
            a.d2_vs_cpu.short(),
            a.d2_vs_gpu.short(),
            a.cpu_vs_gpu.short(),
            a.cold_slots_vs_gpu.short(),
            a.hot_slots_vs_cpu.short(),
        );
        println!(
            "        bitwise mismatching elements: twin {} hot-slots {} cold-slots {} hot-rows {} cold-rows {}  non-finite {}",
            a.twin_mismatch, a.hot_slot_mismatch, a.cold_slot_mismatch, a.hot_row_mismatch, a.cold_row_mismatch, a.nonfinite
        );
        println!(
            "        return transport: mapped calls {} copy calls {}  output_copy_calls {} mapped_publications {} mapped_retire_waits {}  violations {}",
            a.mapped_calls, a.copy_calls, a.output_copy_calls, a.mapped_publications, a.mapped_retire_waits, a.transport_violations
        );
    }
}

fn layer_json(layer: usize, a: &LayerAgg) -> Value {
    json!({
        "layer": layer, "calls": a.calls, "rows": a.rows,
        "hot_slots": a.hot_slots, "cold_slots": a.cold_slots, "invalid_slots": a.invalid_slots,
        "rows_all_hot": a.rows_hot, "rows_all_cold": a.rows_cold, "rows_mixed": a.rows_mixed, "rows_empty": a.rows_empty,
        "d2_vs_all_cpu": a.d2_vs_cpu.json(), "d2_vs_all_gpu": a.d2_vs_gpu.json(), "all_cpu_vs_all_gpu": a.cpu_vs_gpu.json(),
        "cold_slots_vs_gpu_arith": a.cold_slots_vs_gpu.json(), "hot_slots_vs_cpu_arith": a.hot_slots_vs_cpu.json(),
        "bitwise_mismatch_elements": {
            "combine_twin_vs_gpu_combine": a.twin_mismatch, "hot_slots_vs_all_gpu": a.hot_slot_mismatch,
            "cold_slots_vs_all_cpu": a.cold_slot_mismatch, "all_hot_rows_vs_all_gpu": a.hot_row_mismatch,
            "all_cold_rows_vs_all_cpu": a.cold_row_mismatch,
        },
        "nonfinite": a.nonfinite,
        "placement_hot_experts": a.placement_hot,
        "return_transport": {
            "return_mode": if a.mapped_calls > 0 && a.copy_calls == 0 { "mapped" } else if a.mapped_calls == 0 { "copy" } else { "mixed" },
            "mapped_calls": a.mapped_calls, "copy_calls": a.copy_calls,
            "output_copy_calls": a.output_copy_calls, "mapped_publications": a.mapped_publications,
            "mapped_retire_waits": a.mapped_retire_waits, "violations": a.transport_violations,
        },
    })
}

/// A full-capture pass: every record analyzed against its own inputs.
fn full_pass(
    m: &mut Model,
    args: &Args,
    spec: &PassSpec,
    tokens: &[u32],
    layers: &[usize],
) -> R<(BTreeMap<usize, LayerAgg>, PassOut)> {
    let shadow = m.shadow.clone();
    let mut aggs: BTreeMap<usize, LayerAgg> =
        layers.iter().map(|&l| (l, LayerAgg::default())).collect();
    let out = run_pass(m, args, spec, tokens, layers, |_, recs| {
        for rec in &recs {
            analyze(
                rec,
                &shadow,
                aggs.get_mut(&rec.layer)
                    .ok_or("record for an unselected layer")?,
            )?;
        }
        Ok(())
    })?;
    Ok((aggs, out))
}

fn bitwise_gates(gates: &mut Vec<Gate>, run: &str, aggs: &BTreeMap<usize, LayerAgg>, steps: usize) {
    let sum = |f: fn(&LayerAgg) -> u64| aggs.values().map(f).sum::<u64>();
    gate(
        gates,
        format!("{run}: records"),
        aggs.values().all(|a| a.calls == steps as u64),
        format!("{} layers x {steps} steps, one call each", aggs.len()),
    );
    gate(
        gates,
        format!("{run}: return transport accounting"),
        sum(|a| a.transport_violations) == 0,
        format!(
            "mapped calls {} copy calls {}, output_copy_calls {} mapped_publications {} mapped_retire_waits {} (a mapped call publishes once and issues zero cold-slot copies; a copy call publishes nothing)",
            sum(|a| a.mapped_calls), sum(|a| a.copy_calls), sum(|a| a.output_copy_calls),
            sum(|a| a.mapped_publications), sum(|a| a.mapped_retire_waits)
        ),
    );
    gate(
        gates,
        format!("{run}: finite"),
        sum(|a| a.nonfinite) == 0,
        format!("{} non-finite values", sum(|a| a.nonfinite)),
    );
    gate(
        gates,
        format!("{run}: host canonical combine == GPU combine (twin)"),
        sum(|a| a.twin_mismatch) == 0,
        format!("{} mismatching elements", sum(|a| a.twin_mismatch)),
    );
    gate(
        gates,
        format!("{run}: hot slots bitwise vs all-GPU tail"),
        sum(|a| a.hot_slot_mismatch) == 0,
        format!(
            "{} slots, {} mismatching elements",
            sum(|a| a.hot_slots),
            sum(|a| a.hot_slot_mismatch)
        ),
    );
    gate(
        gates,
        format!("{run}: cold slots bitwise vs all-CPU D1 arithmetic"),
        sum(|a| a.cold_slot_mismatch) == 0,
        format!(
            "{} slots, {} mismatching elements",
            sum(|a| a.cold_slots),
            sum(|a| a.cold_slot_mismatch)
        ),
    );
    gate(
        gates,
        format!("{run}: all-hot rows bitwise vs all-GPU target"),
        sum(|a| a.hot_row_mismatch) == 0,
        format!(
            "{} rows, {} mismatching elements",
            sum(|a| a.rows_hot),
            sum(|a| a.hot_row_mismatch)
        ),
    );
    gate(
        gates,
        format!("{run}: all-cold rows bitwise vs all-CPU target"),
        sum(|a| a.cold_row_mismatch) == 0,
        format!(
            "{} rows, {} mismatching elements",
            sum(|a| a.rows_cold),
            sum(|a| a.cold_row_mismatch)
        ),
    );
}

/// Hex of a layer bitmap exactly as the core identity renders it: bit `i` is layer `i`, one
/// big number, most significant byte first, `"0"` when empty.
fn layer_bitmap_hex(labels: &[usize]) -> String {
    let Some(&max) = labels.iter().max() else {
        return "0".to_string();
    };
    let mut bytes = vec![0u8; max / 8 + 1];
    for &label in labels {
        bytes[label / 8] |= 1 << (label % 8);
    }
    bytes.iter().rev().map(|b| format!("{b:02x}")).collect()
}

/// The identity a pure whole-layer split over exactly `armed` must carry.
fn whole_identity(armed: &[usize]) -> String {
    format!("cpu-whole-split-v1/{}", layer_bitmap_hex(armed))
}

fn real_main() -> R<bool> {
    let args = parse_args()?;
    let text = std::fs::read_to_string(&args.text)
        .map_err(|x| format!("read {}: {x}", args.text.display()))?;
    let need_tokens = args.offset + args.prompt_rows + args.steps * args.rows;
    let max_seq = need_tokens.next_multiple_of(256).max(512);
    let mut m = load(&args, max_seq)?;
    let stream = m.tokenizer.encode(&text);
    if stream.len() < need_tokens {
        return Err(format!(
            "TEXT has {} tokens, --offset {} + --prompt-rows {} + --steps {} x --rows {} needs {need_tokens}",
            stream.len(),
            args.offset,
            args.prompt_rows,
            args.steps,
            args.rows
        ));
    }
    let tokens: Vec<u32> = stream[args.offset..need_tokens].to_vec();

    let armed = m.bundle.split_layers();
    if armed.is_empty() {
        return Err(if args.untiered {
            "the whole-layer D2 split armed on no layer: needs HIPFIRE_QWEN4_D2_DEFAULT=1 with HIPFIRE_OFFLOAD_EXEC unset, aggregate QT44/QT53 experts host-mapped (the product's automatic placement, or a diagnostic HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=N), and HIPFIRE_MOE_CPU_EXPERTS != 0"
                .into()
        } else {
            "the D2 split armed on no layer: needs HIPFIRE_OFFLOAD_EXEC=cpu, the hot set applied (explicit sidecar on gfx1100) and host-tier experts with CPU spans"
                .into()
        });
    }
    let layers: Vec<usize> = match &args.layers {
        LayerSel::All => armed.clone(),
        LayerSel::First(n) => armed.iter().copied().take(*n).collect(),
        LayerSel::List(list) => {
            if let Some(bad) = list.iter().find(|l| !armed.contains(l)) {
                return Err(format!(
                    "--layers: layer {bad} is not armed for the split (armed: {armed:?})"
                ));
            }
            list.clone()
        }
    };
    if layers.is_empty() {
        return Err("--layers selected no layer".into());
    }
    let digest = m.bundle.split_digest();
    let identity = m.bundle.host_exec_identity();
    let owner_kinds = m.bundle.split_owner_kinds();
    let shape = if args.untiered {
        format!(
            "UN-TIERED whole-layer ({} host-placed layers {:?}, placement {}, no hot set, D1 span digest {})",
            m.host_layers.len(),
            m.host_layers,
            &m.placement_digest[..16],
            &m.span_digest[..16]
        )
    } else {
        format!(
            "hot set {} experts (plan {})",
            m.hot_experts,
            &m.plan_digest[..16]
        )
    };
    println!(
        "qwen4_cpu_split_parity: {} armed layers, {} selected {:?}, {shape}, split digest {}, identity {identity}, {} threads, prompt {} rows, {} steps x {} rows, {}",
        armed.len(),
        layers.len(),
        layers,
        digest.as_deref().unwrap_or("-"),
        std::env::var("HIPFIRE_CPU_EXPERT_THREADS").unwrap_or_else(|_| "default".into()),
        args.prompt_rows,
        args.steps,
        args.rows,
        if args.greedy { "greedy" } else { "teacher-forced" },
    );
    println!(
        "qwen4_cpu_split_parity: HIPFIRE_QWEN4_D2_RETURN requested {} (a mixed-placement layer keeps copy; per-layer return mode and counters are reported below)",
        hipfire_config::memory::qwen4_d2_transport(&m.gpu.arch)
            .map_or_else(|refusal| format!("INVALID ({refusal})"), |t| t.ret.as_str().to_string()),
    );
    // The lab parity modes drain the stream before the CPU arithmetic and
    // serialize the combine, which a CP-wait join cannot do: refuse instead of
    // silently measuring something else.
    let ready_modes = m.bundle.split_ready_modes();
    let join_modes = m.bundle.split_join_modes();
    let event_layers = ready_modes
        .iter()
        .filter(|(_, mode)| *mode == "event")
        .count();
    let cpwait_layers = join_modes
        .iter()
        .filter(|(_, mode)| *mode == "cpwait")
        .count();
    println!(
        "qwen4_cpu_split_parity: HIPFIRE_QWEN4_D2_READY requested {} ({event_layers} of {} armed layers run event readiness); HIPFIRE_QWEN4_D2_JOIN requested {} ({cpwait_layers} run the CP-wait join; refused layers keep host)",
        hipfire_config::memory::qwen4_d2_ready()
            .map_or_else(|refusal| format!("INVALID ({refusal})"), |mode| mode.as_str().to_string()),
        ready_modes.len(),
        hipfire_config::memory::qwen4_d2_transport(&m.gpu.arch)
            .map_or_else(|refusal| format!("INVALID ({refusal})"), |t| t.join.as_str().to_string()),
    );
    if cpwait_layers > 0 {
        return Err(format!(
            "HIPFIRE_QWEN4_D2_JOIN=cpwait is active on {cpwait_layers} layers; the lab parity runs drain the stream before the join and cannot run under it (run with HIPFIRE_QWEN4_D2_JOIN=host; exact gfx1100 defaults an unset join to cpwait)"
        ));
    }

    let mut gates: Vec<Gate> = Vec::new();
    let mut runs_json: Vec<Value> = Vec::new();

    // ── identity: owner kind, identity string, placement, lab force ──
    let want_kind = if args.untiered { "whole-layer" } else { "tier" };
    gate(
        &mut gates,
        format!("identity: every armed layer is a {want_kind} owner"),
        owner_kinds.iter().map(|(l, _)| *l).collect::<Vec<_>>() == armed
            && owner_kinds.iter().all(|(_, k)| *k == want_kind),
        format!(
            "{} armed layers, owner kinds {:?}",
            armed.len(),
            owner_kinds.iter().map(|(_, k)| *k).collect::<BTreeSet<_>>()
        ),
    );
    if args.untiered {
        let want = whole_identity(&armed);
        gate(
            &mut gates,
            "identity: host_exec_identity == cpu-whole-split-v1/<armed bitmap> (no D1/tier part)",
            identity == want && digest.as_deref() == Some(want.as_str()),
            format!("identity {identity}, split_digest {digest:?}, expected {want}"),
        );
        let host_layers = m.host_layers.clone();
        gate(
            &mut gates,
            "identity: armed layers == every resolved host-placed trunk layer",
            armed == host_layers,
            format!(
                "armed {} layers {armed:?}, resolved host-placed layers {host_layers:?}, placement digest {}",
                armed.len(),
                m.placement_digest
            ),
        );
        let (force, explicit) = (
            hipfire_config::memory::qwen4_d2_default(),
            hipfire_config::memory::offload_exec_explicit(),
        );
        gate(
            &mut gates,
            "lab force: developer.qwen4_d2_default=true with memory.offload_exec unset",
            force == Some(true) && explicit.is_none(),
            format!("qwen4_d2_default {force:?}, offload_exec explicit {explicit:?}"),
        );
    } else {
        gate(
            &mut gates,
            "identity: tier identity unchanged (cpu-tier-split-v1/..)",
            identity.starts_with("cpu-tier-split-v1/")
                && !identity.contains("cpu-whole-split-v1")
                && digest.as_deref() == Some(identity.as_str()),
            format!("identity {identity}"),
        );
    }

    // ── natural placement: same-input comparisons ────────────────────
    let natural = PassSpec {
        name: "natural".into(),
        mode: SplitParityMode::Full,
        endpoint: SplitEndpoint::Natural,
        schedule: Schedule::Overlapped,
    };
    println!(
        "RUN natural ({}, overlapped schedule, full capture)",
        if args.untiered {
            "un-tiered: every valid slot cold"
        } else {
            "real mixed placement"
        }
    );
    let (aggs, natural_out) = full_pass(&mut m, &args, &natural, &tokens, &layers)?;
    print_layers(&aggs);
    bitwise_gates(&mut gates, "natural", &aggs, args.steps);
    let hot_total: u64 = aggs.values().map(|a| a.hot_slots).sum();
    let cold_total: u64 = aggs.values().map(|a| a.cold_slots).sum();
    if args.untiered {
        let placement_hot: u64 = aggs.values().map(|a| a.placement_hot).sum();
        gate(
            &mut gates,
            "natural: all-cold placement (no hot slot, no device expert in the mask)",
            hot_total == 0 && placement_hot == 0 && cold_total > 0,
            format!("{hot_total} hot and {cold_total} cold slots, {placement_hot} device experts in the placement masks"),
        );
    } else {
        gate(
            &mut gates,
            "natural: join exercised",
            hot_total > 0 && cold_total > 0,
            format!("{hot_total} hot and {cold_total} cold slots in the selected layers"),
        );
    }
    let worst = |f: fn(&LayerAgg) -> &Diff| {
        let mut d = Diff::default();
        aggs.values().for_each(|a| d.merge(f(a)));
        d
    };
    let (w_cpu, w_gpu, w_ref) = (
        worst(|a| &a.d2_vs_cpu),
        worst(|a| &a.d2_vs_gpu),
        worst(|a| &a.cpu_vs_gpu),
    );
    println!(
        "  WORST D2 vs all-CPU {} (rel L2 {:.3e}), D2 vs all-GPU {} (rel L2 {:.3e}), all-CPU vs all-GPU {} (rel L2 {:.3e})",
        w_cpu.short(), w_cpu.rel_l2(), w_gpu.short(), w_gpu.rel_l2(), w_ref.short(), w_ref.rel_l2()
    );
    runs_json.push(json!({
        "run": "natural", "layers": aggs.iter().map(|(l, a)| layer_json(*l, a)).collect::<Vec<_>>(),
        "worst": {"d2_vs_all_cpu": w_cpu.json(), "d2_vs_all_gpu": w_gpu.json(), "all_cpu_vs_all_gpu": w_ref.json()},
        "logits_sha256": natural_out.digests,
    }));

    // ── endpoints ────────────────────────────────────────────────────
    if args.endpoints {
        for (name, endpoint) in [
            ("all-hot", SplitEndpoint::AllHot),
            ("all-cold", SplitEndpoint::AllCold),
        ] {
            println!("RUN {name} (placement endpoint forced inside the executor, full capture)");
            let spec = PassSpec {
                name: name.into(),
                mode: SplitParityMode::Full,
                endpoint,
                schedule: Schedule::Overlapped,
            };
            let (aggs, out) = full_pass(&mut m, &args, &spec, &tokens, &layers)?;
            print_layers(&aggs);
            bitwise_gates(&mut gates, name, &aggs, args.steps);
            let valid_rows: u64 = aggs.values().map(|a| a.rows).sum();
            let (side_rows, side) = match endpoint {
                SplitEndpoint::AllHot => {
                    (aggs.values().map(|a| a.rows_hot).sum::<u64>(), "all-GPU")
                }
                _ => (aggs.values().map(|a| a.rows_cold).sum::<u64>(), "all-CPU"),
            };
            let empty: u64 = aggs.values().map(|a| a.rows_empty).sum();
            gate(
                &mut gates,
                format!("{name}: endpoint D2 target bitwise == {side} target on every row"),
                side_rows + empty == valid_rows,
                format!("{side_rows} of {valid_rows} rows ({empty} unrouted) compared bitwise; mismatching rows would fail the row gates above"),
            );
            if args.untiered && endpoint == SplitEndpoint::AllCold {
                // The natural run is the all-cold placement computed from the owners' pinned
                // bytes; this run computes every slot from the artifact's D1 spans. Same
                // arithmetic on equal bytes: the logits must agree bit for bit.
                let first = first_digest_mismatch(&out.digests, &natural_out.digests);
                gate(
                    &mut gates,
                    "all-cold: logits bit-identical to the natural run (owner bytes == D1 span bytes)",
                    first.is_none(),
                    format!("{} steps, first mismatching step {first:?}", args.steps),
                );
            }
            runs_json.push(json!({
                "run": name, "layers": aggs.iter().map(|(l, a)| layer_json(*l, a)).collect::<Vec<_>>(),
                "logits_sha256": out.digests,
            }));
        }
    }

    // ── serial vs overlapped, identical kernels ──────────────────────
    if args.timing {
        println!(
            "RUN timing: serial (GPU tail drained, then CPU slots) vs overlapped, ABBA x {} rounds, steps >= {} counted",
            args.timing_rounds, args.warmup_steps
        );
        let mut over: BTreeMap<usize, TimingAgg> =
            layers.iter().map(|&l| (l, TimingAgg::default())).collect();
        let mut serial: BTreeMap<usize, TimingAgg> =
            layers.iter().map(|&l| (l, TimingAgg::default())).collect();
        let mut digests_ok = true;
        let mut mismatched = Vec::new();
        let mut timing_runs = Vec::new();
        let mut timing_runs_json = Vec::new();
        for round in 0..args.timing_rounds {
            let order = if round % 2 == 0 {
                [Schedule::Overlapped, Schedule::Serial]
            } else {
                [Schedule::Serial, Schedule::Overlapped]
            };
            for schedule in order {
                let spec = PassSpec {
                    name: format!(
                        "timing-{}-r{round}",
                        if schedule == Schedule::Serial {
                            "serial"
                        } else {
                            "overlapped"
                        }
                    ),
                    mode: SplitParityMode::Timing,
                    endpoint: SplitEndpoint::Natural,
                    schedule,
                };
                let warmup = args.warmup_steps;
                let target = if schedule == Schedule::Serial {
                    &mut serial
                } else {
                    &mut over
                };
                let out = run_pass(&mut m, &args, &spec, &tokens, &layers, |step, recs| {
                    if step >= warmup {
                        for rec in &recs {
                            target
                                .get_mut(&rec.layer)
                                .ok_or("record for an unselected layer")?
                                .add(&rec.timing);
                        }
                    }
                    Ok(())
                })?;
                if out.digests != natural_out.digests {
                    digests_ok = false;
                    mismatched.push(spec.name.clone());
                }
                let first = first_digest_mismatch(&out.digests, &natural_out.digests);
                timing_runs_json.push(json!({
                    "run": spec.name, "round": round,
                    "serial": schedule == Schedule::Serial,
                    "schedule": if schedule == Schedule::Serial { "serial" } else { "overlapped" },
                    "logits_sha256": out.digests,
                    "matching_natural": first.is_none(), "first_mismatch_step": first,
                }));
                timing_runs.push(TimingRun {
                    name: spec.name,
                    round,
                    schedule,
                    digests: out.digests,
                });
            }
        }
        println!(
            "  {:>5} {:>13} {:>13} {:>8} | overlapped: {:>10} {:>10} {:>10} | serial: {:>10} {:>10} {:>10}  (median ns)",
            "layer", "wall ovl", "wall ser", "ser/ovl", "enqueue", "cpu", "tail", "enqueue", "cpu", "tail"
        );
        let mut layer_rows = Vec::new();
        let (mut all_o, mut all_s) = (Vec::new(), Vec::new());
        for &layer in &layers {
            let (o, s) = (&over[&layer], &serial[&layer]);
            let (ow, sw) = (o.wall_median(), s.wall_median());
            println!(
                "  {:>5} {:>13} {:>13} {:>8.3} | overlapped: {:>10} {:>10} {:>10} | serial: {:>10} {:>10} {:>10}",
                layer,
                ow,
                sw,
                if ow == 0 { 0.0 } else { sw as f64 / ow as f64 },
                median(&mut o.enqueue.clone()),
                median(&mut o.cpu.clone()),
                median(&mut o.tail.clone()),
                median(&mut s.enqueue.clone()),
                median(&mut s.cpu.clone()),
                median(&mut s.tail.clone()),
            );
            all_o.extend_from_slice(&o.wall);
            all_s.extend_from_slice(&s.wall);
            layer_rows.push(json!({"layer": layer, "overlapped": o.stats(), "serial": s.stats()}));
        }
        let (om, sm) = (median(&mut all_o.clone()), median(&mut all_s.clone()));
        println!(
            "  POOLED per-layer wall median: overlapped {om} ns, serial {sm} ns, serial/overlapped {:.3}",
            if om == 0 { 0.0 } else { sm as f64 / om as f64 }
        );
        gate(
            &mut gates,
            "timing: logits bit-identical across natural full / overlapped / serial runs",
            digests_ok,
            if digests_ok {
                format!("{} steps x {} runs", args.steps, 1 + 2 * args.timing_rounds)
            } else {
                format!("runs differing from the natural full run: {mismatched:?}")
            },
        );
        let (repeat_comparisons, repeats_match) = timing_comparisons(&timing_runs, true);
        let (schedule_comparisons, schedules_match) = timing_comparisons(&timing_runs, false);
        let repeats_checked = [Schedule::Overlapped, Schedule::Serial]
            .iter()
            .all(|schedule| {
                timing_runs
                    .iter()
                    .filter(|r| r.schedule == *schedule)
                    .map(|r| r.round)
                    .collect::<BTreeSet<_>>()
                    .len()
                    >= 2
            });
        let repeat_status = if !repeats_match {
            "fail"
        } else if repeats_checked {
            "pass"
        } else {
            "not_checked"
        };
        if repeats_checked || !repeats_match {
            gate(
                &mut gates,
                "timing: same-schedule repeat logits bit-identical",
                repeats_match,
                format!(
                    "{} same-schedule comparisons; see timing.repeat_comparisons",
                    repeat_comparisons.len()
                ),
            );
        } else {
            println!("  GATE NOT_CHECKED timing: same-schedule repeat logits bit-identical: requires at least two rounds per schedule");
        }
        gate(
            &mut gates,
            "timing: samples",
            over.values()
                .chain(serial.values())
                .all(|t| t.n as usize == args.timing_rounds * (args.steps - args.warmup_steps)),
            format!(
                "{} calls per layer per schedule",
                args.timing_rounds * (args.steps - args.warmup_steps)
            ),
        );
        runs_json.push(json!({
            "run": "timing", "rounds": args.timing_rounds, "warmup_steps": args.warmup_steps,
            "runs": timing_runs_json,
            "repeat_gate": {
                "status": repeat_status,
                "pass": if repeat_status == "not_checked" { None } else { Some(repeats_match) },
                "required_runs_per_schedule": 2,
            },
            "repeat_comparisons": repeat_comparisons,
            "serial_vs_overlapped": {
                "status": if schedule_comparisons.is_empty() { "not_checked" } else if schedules_match { "pass" } else { "fail" },
                "comparisons": schedule_comparisons,
            },
            "layers": layer_rows,
            "pooled": {"overlapped_wall_median_ns": om, "serial_wall_median_ns": sm},
        }));
    }

    // Cumulative readiness/join counters of this process (every run above).
    let counters = m.bundle.split_readiness_counters();
    let counter_sum = |i: usize| counters.iter().map(|(_, c)| c[i]).sum::<u64>();
    println!(
        "qwen4_cpu_split_parity: readiness counters over {} layers: ready_polls {}, ready_fallback_syncs {}, cpwait_joins {}, cpwait_aborts {}, watchdog_fires {}",
        counters.len(),
        counter_sum(0),
        counter_sum(1),
        counter_sum(2),
        counter_sum(3),
        counter_sum(4),
    );
    if event_layers > 0 {
        gate(
            &mut gates,
            "readiness: event layers polled their input proof",
            counter_sum(0) > 0,
            format!(
                "{event_layers} event layers, {} polls, {} fallback syncs",
                counter_sum(0),
                counter_sum(1)
            ),
        );
    }
    let all_ok = gates.iter().all(|g| g.ok);
    println!(
        "SUMMARY {} ({} gates, {} failed)",
        if all_ok { "PASS" } else { "FAIL" },
        gates.len(),
        gates.iter().filter(|g| !g.ok).count()
    );
    if let Some(path) = &args.json {
        let doc = json!({
            "tool": "qwen4_cpu_split_parity",
            "model": args.model.display().to_string(),
            "text": args.text.display().to_string(),
            "arch": m.gpu.arch.to_string(),
            "hot_budget_mib": args.hot_budget_mib,
            "untiered": args.untiered,
            "untiered_vram_layers_diagnostic": m.untiered_vram_layers,
            "host_placed_trunk_layers": m.host_layers,
            "placement_digest": m.placement_digest,
            "placement_source": m.placement_source,
            "d2_force": hipfire_config::memory::qwen4_d2_default(),
            "d2_return_requested": hipfire_config::memory::qwen4_d2_transport(&m.gpu.arch).ok().map(|t| t.ret.as_str()),
            "d2_ready_requested": hipfire_config::memory::qwen4_d2_ready().ok().map(|mode| mode.as_str()),
            "d2_join_requested": hipfire_config::memory::qwen4_d2_transport(&m.gpu.arch).ok().map(|t| t.join.as_str()),
            "d2_ready_modes": ready_modes.iter().map(|(l, k)| json!([l, k])).collect::<Vec<_>>(),
            "d2_join_modes": join_modes.iter().map(|(l, k)| json!([l, k])).collect::<Vec<_>>(),
            "d2_readiness_counters": counters.iter().map(|(l, c)| json!({
                "layer": l, "ready_polls": c[0], "ready_fallback_syncs": c[1],
                "cpwait_joins": c[2], "cpwait_aborts": c[3], "watchdog_fires": c[4],
            })).collect::<Vec<_>>(),
            "offload_exec_explicit": hipfire_config::memory::offload_exec_explicit().map(|v| v.to_string()),
            "span_digest": m.span_digest,
            "host_exec_identity": identity,
            "split_owner_kinds": owner_kinds.iter().map(|(l, k)| json!([l, k])).collect::<Vec<_>>(),
            "hot_experts": m.hot_experts,
            "plan_digest": m.plan_digest,
            "split_digest": digest,
            "armed_layers": armed,
            "selected_layers": layers,
            "prompt_rows": args.prompt_rows, "steps": args.steps, "rows": args.rows, "greedy": args.greedy,
            "pass": all_ok,
            "gates": gates.iter().map(|g| json!({"name": g.name, "pass": g.ok, "detail": g.detail})).collect::<Vec<_>>(),
            "runs": runs_json,
        });
        std::fs::write(path, serde_json::to_vec_pretty(&doc).map_err(e)?)
            .map_err(|x| format!("write {}: {x}", path.display()))?;
    }
    drop(m);
    Ok(all_ok)
}

fn main() {
    match real_main() {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(m) => {
            eprintln!("qwen4_cpu_split_parity: {m}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hashes(values: &[&str]) -> Vec<String> {
        values.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn first_digest_divergence_and_length_mismatch() {
        assert_eq!(
            first_digest_mismatch(&hashes(&["a", "b"]), &hashes(&["a", "b"])),
            None
        );
        assert_eq!(
            first_digest_mismatch(&hashes(&["a", "b"]), &hashes(&["x", "b"])),
            Some(0)
        );
        assert_eq!(
            first_digest_mismatch(&hashes(&["a", "b"]), &hashes(&["a", "x"])),
            Some(1)
        );
        assert_eq!(
            first_digest_mismatch(&hashes(&["a"]), &hashes(&["a", "b"])),
            Some(1)
        );
        assert_eq!(
            first_digest_mismatch(&hashes(&["a", "b"]), &hashes(&["a"])),
            Some(1)
        );
        assert_eq!(first_digest_mismatch(&[], &hashes(&["a"])), Some(0));
        assert_eq!(first_digest_mismatch(&[], &[]), None);
    }

    #[test]
    fn repeat_comparison_separates_schedules() {
        let mut runs = Vec::new();
        for round in 0..2 {
            for (schedule, hash) in [(Schedule::Overlapped, "o"), (Schedule::Serial, "s")] {
                runs.push(TimingRun {
                    name: format!("run-{}", runs.len()),
                    round,
                    schedule,
                    digests: hashes(&[hash]),
                });
            }
        }
        let (repeats, matching) = timing_comparisons(&runs, true);
        assert!(matching);
        assert_eq!(repeats.len(), 2);
        let (cross, matching) = timing_comparisons(&runs, false);
        assert!(!matching);
        assert_eq!(cross.len(), 4);
        runs[2].digests.push("extra".into());
        let (repeats, matching) = timing_comparisons(&runs, true);
        assert!(!matching);
        assert_eq!(repeats[0]["first_mismatch_step"], json!(1));
        let (pairs, matching) = timing_comparisons(&runs[..2], true);
        assert!(pairs.is_empty());
        assert!(matching); // No evidence: caller must report not_checked, not pass.
    }

    #[test]
    fn whole_identity_renders_the_core_bitmap() {
        assert_eq!(layer_bitmap_hex(&[]), "0");
        assert_eq!(
            whole_identity(&(0..48).collect::<Vec<_>>()),
            "cpu-whole-split-v1/ffffffffffff"
        );
        assert_eq!(whole_identity(&[0]), "cpu-whole-split-v1/01");
        assert_eq!(whole_identity(&[9]), "cpu-whole-split-v1/0200");
        assert_eq!(whole_identity(&[1, 9]), "cpu-whole-split-v1/0202");
    }

    #[test]
    fn untiered_admission_requires_force_and_no_explicit_exec() {
        use hipfire_config::memory::OffloadExec;
        assert!(admit_untiered(None, Some(true), false).is_ok());
        assert!(admit_untiered(None, None, false).is_err());
        assert!(admit_untiered(None, Some(false), false).is_err());
        assert!(admit_untiered(None, Some(true), true).is_err());
        for mode in [OffloadExec::Pcie, OffloadExec::Cpu, OffloadExec::Passback] {
            assert!(
                admit_untiered(Some(mode), Some(true), false).is_err(),
                "{mode}"
            );
        }
    }

    #[test]
    fn untiered_fixed_layers_are_an_optional_override() {
        use hipfire_arch_qwen4::expert_residency::ExpertVramLayers;
        assert_eq!(
            untiered_vram_layers(Some(ExpertVramLayers::Layers(0)), 48),
            Ok(Some(0))
        );
        assert_eq!(
            untiered_vram_layers(Some(ExpertVramLayers::Layers(47)), 48),
            Ok(Some(47))
        );
        assert!(untiered_vram_layers(Some(ExpertVramLayers::Layers(48)), 48).is_err());
        // Unset and `auto` both resolve the product's automatic placement.
        assert_eq!(
            untiered_vram_layers(Some(ExpertVramLayers::Auto), 48),
            Ok(None)
        );
        assert_eq!(untiered_vram_layers(None, 48), Ok(None));
    }
}
