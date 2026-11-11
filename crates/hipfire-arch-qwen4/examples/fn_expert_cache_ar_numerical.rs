//! Exact AR expert-cache numerical oracle through the production rank loader.
//! No MTP head or speculative reserve is loaded.
//! Hashes every full-vocabulary F32 logit bit and canonical active target-state
//! byte at each AR step; this is not a serving fast-head interchangeability gate.
//! INTRUSIVE NUMERICAL ONLY: canonical_digest_rows calls device_synchronize,
//! draining the copy stream as well. No async-overlap or performance proof.
//! Short-prefill counters prove absence of additional manager bulk restoration,
//! not that pending DMA remained physically in flight during that prefill.
use hipfire_arch_qwen4::{bundle::Qwen4Bundle, Qwen4KvBackend};
use hipfire_arch_qwen4::{
    expert_tier_owners::{ExpertReadView, ExpertTierLease},
    expert_tier_plan::Placement,
};
use hipfire_loader::{load_qwen4_rank, Qwen4RankShared};
use hipfire_runtime::{
    loader_api::{ModelSource, SpecLoadCfg},
    tokenizer::Tokenizer,
};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, path::PathBuf};

type R<T> = Result<T, String>;
fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}
fn require(ok: bool, why: impl Into<String>) -> R<()> {
    if ok {
        Ok(())
    } else {
        Err(why.into())
    }
}
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn ids_sha(ids: &[u32]) -> String {
    sha(&ids.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
}

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    id: String,
    prompt_file: PathBuf,
    #[serde(default = "prompt_rows")]
    prompt_rows: usize,
    // Deterministic suffixes cycle these prompt IDs, with a fixed offset.
    // Synthetic suffixes measure exactness, not task quality.
    #[serde(default = "short_rows")]
    short_rows: usize,
    #[serde(default = "staged_rows")]
    staged_rows: usize,
}
fn prompt_rows() -> usize {
    256
}
fn short_rows() -> usize {
    128
}
fn staged_rows() -> usize {
    5103
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
struct Point {
    case: String,
    phase: String,
    step: usize,
    position: usize,
    history_sha256: String,
    vocab: usize,
    logits_bytes: usize,
    logits_sha256: String,
    top1: u32,
    // Canonical keys, semantic metadata, and SHA256 over exactly ACTIVE bytes.
    // No unused capacity, process IDs, allocation pointers or snapshot scratch.
    state: Vec<(String, String)>,
}
/// The actual HFQ container/index binds source offsets, byte lengths, dtypes,
/// shapes and metadata. No payload reread or physical destination address.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
struct PartitionSource {
    artifact_bytes: u64,
    artifact_index_sha256: String,
    layers: u32,
    experts_per_layer: u32,
}
#[derive(Clone)]
struct SemanticExpert {
    layer: u32,
    expert: u32,
    original_device: bool,
    gate_up_qtype: u8,
    down_qtype: u8,
    gate_up_bytes: u64,
    down_bytes: u64,
}
fn semantic_partition_digest(source: &PartitionSource, rows: &[SemanticExpert]) -> R<String> {
    require(
        source.layers > 0 && source.experts_per_layer > 0,
        "empty partition geometry",
    )?;
    require(
        source.artifact_index_sha256.len() == 64,
        "missing actual artifact index identity",
    )?;
    let expected = u64::from(source.layers)
        .checked_mul(u64::from(source.experts_per_layer))
        .and_then(|n| usize::try_from(n).ok())
        .ok_or("partition extent overflow")?;
    require(
        rows.len() == expected,
        "missing original expert partition rows",
    )?;
    let mut digest = Sha256::new();
    digest.update(b"hipfire.fn.original-expert-partition.v1");
    digest.update(serde_json::to_vec(source).map_err(err)?);
    for (index, row) in rows.iter().enumerate() {
        let layer = (index / source.experts_per_layer as usize) as u32;
        let expert = (index % source.experts_per_layer as usize) as u32;
        require(
            (row.layer, row.expert) == (layer, expert),
            "duplicate, reordered or missing original expert",
        )?;
        require(
            row.gate_up_bytes > 0 && row.down_bytes > 0,
            "empty original projection",
        )?;
        digest.update(row.layer.to_le_bytes());
        digest.update(row.expert.to_le_bytes());
        digest.update([
            u8::from(row.original_device),
            row.gate_up_qtype,
            row.down_qtype,
        ]);
        digest.update(row.gate_up_bytes.to_le_bytes());
        digest.update(row.down_bytes.to_le_bytes());
    }
    Ok(format!("{:x}", digest.finalize()))
}
fn actual_semantic_partition(
    lease: &ExpertTierLease,
    source: &PartitionSource,
) -> R<(String, u64)> {
    let mut rows = Vec::new();
    let mut resident = 0u64;
    for layer in 0..source.layers {
        // Resident is the original frozen pin. Cache donor mirrors appear in
        // BorrowPrefill/canonical cache views, not this original residency map.
        let view = lease
            .layer_view(layer as usize, ExpertReadView::Resident)
            .ok_or("missing original resident layer view")?;
        for expert in 0..source.experts_per_layer {
            let entry = view
                .entry(expert as usize)
                .ok_or("missing original expert")?;
            let original_device = entry.tier == Placement::Device;
            resident += u64::from(original_device);
            let (gate_up_qtype, down_qtype) = entry.format.pair();
            rows.push(SemanticExpert {
                layer,
                expert,
                original_device,
                gate_up_qtype,
                down_qtype,
                gate_up_bytes: entry.gate_up_len,
                down_bytes: entry.down_len,
            });
        }
    }
    Ok((semantic_partition_digest(source, &rows)?, resident))
}
#[derive(Clone, Debug, Deserialize, Serialize)]
struct Report {
    schema: String,
    requested_cache: bool,
    actual_cache: bool,
    native_mtp: bool,
    max_seq: usize,
    decode_steps: usize,
    fixture_sha256: String,
    hotset_sha256: String,
    // Deliberately recorded, not compared: mirrors change this physical plan.
    physical_plan_sha256: String,
    // Compared exactly: original pin + source identity/geometry + codecs/bytes.
    semantic_partition_sha256: String,
    source_identity: PartitionSource,
    state_format: String,
    kv_backend: String,
    points: Vec<Point>,
    coverage: Value,
    identity: Value,
    performance_claim: bool,
}

fn check_structure(report: &Report) -> R<()> {
    require(
        report.schema == "hipfire-fn-ar-lru-numerical-v2",
        "wrong schema",
    )?;
    require(
        report.requested_cache == report.actual_cache,
        "cache arm was not actually enabled",
    )?;
    require(
        !report.native_mtp,
        "AR profile accidentally reserved native MTP",
    )?;
    require(!report.points.is_empty(), "no numerical observations")?;
    for p in &report.points {
        require(
            p.vocab > 0 && p.logits_bytes == p.vocab * 4,
            "truncated full-vocabulary logits",
        )?;
        require(p.logits_sha256.len() == 64, "missing full-vocabulary hash")?;
        require(!p.state.is_empty(), "missing target state")?;
        let keys: BTreeSet<_> = p.state.iter().map(|(k, _)| k).collect();
        require(keys.len() == p.state.len(), "duplicate state key")?;
        require(
            p.state.iter().any(|(k, _)| k.ends_with(".recurrent")),
            "missing GDN recurrent state",
        )?;
        require(
            p.state.iter().any(|(k, _)| k.ends_with(".full_keys")),
            "missing QSA live state",
        )?;
    }
    Ok(())
}
fn record_point(points: &mut Vec<Point>, point: Point, reference: Option<&Report>) -> R<()> {
    if let Some(reference) = reference {
        let index = points.len();
        let expected = reference
            .points
            .get(index)
            .ok_or("reference point sequence truncated")?;
        require(
            expected == &point,
            format!(
                "first numerical mismatch at point {index}, {}/{} step {}",
                point.case, point.phase, point.step
            ),
        )?;
    }
    points.push(point);
    Ok(())
}
fn compare(reference: &Report, candidate: &Report) -> R<()> {
    check_structure(reference)?;
    check_structure(candidate)?;
    require(
        !reference.actual_cache && candidate.actual_cache,
        "pair must be cache-off then cache-on",
    )?;
    require(
        reference.max_seq == candidate.max_seq
            && reference.decode_steps == candidate.decode_steps
            && reference.fixture_sha256 == candidate.fixture_sha256
            && reference.hotset_sha256 == candidate.hotset_sha256
            && reference.semantic_partition_sha256 == candidate.semantic_partition_sha256
            && reference.source_identity == candidate.source_identity
            && reference.state_format == candidate.state_format
            && reference.kv_backend == candidate.kv_backend,
        "paired model/profile/fixture identity differs",
    )?;
    require(
        reference.points.len() == candidate.points.len(),
        "missing or truncated point sequence",
    )?;
    for (n, (a, b)) in reference.points.iter().zip(&candidate.points).enumerate() {
        require(
            a == b,
            format!(
                "point {n} differs: {}/{} step {} (full logits, history or target state)",
                a.case, a.phase, a.step
            ),
        )?;
    }
    Ok(())
}
fn count(census: &Value, name: &str) -> R<u64> {
    census["stats"][name]
        .as_u64()
        .ok_or_else(|| format!("missing live cache counter {name}"))
}
fn delta(before: &Value, after: &Value, name: &str) -> R<u64> {
    count(after, name)?
        .checked_sub(count(before, name)?)
        .ok_or_else(|| format!("counter {name} decreased"))
}
fn census(bundle: &Qwen4Bundle, gpu: &Gpu, cache: bool) -> R<Value> {
    let got = bundle.dev_expert_cache_census(gpu).map_err(err)?;
    match (cache, got) {
        (false, None) => Ok(Value::Null),
        (true, Some(v)) if v["enabled"] == true => Ok(v),
        _ => Err("requested cache arm is absent or unexpectedly present".into()),
    }
}
fn observe(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    logits: &GpuTensor,
    case: &str,
    phase: &str,
    step: usize,
    history: &[u32],
) -> R<Point> {
    let row = gpu.download_f32(logits).map_err(err)?;
    require(
        row.len() == bundle.config.vocab_size && row.iter().all(|x| x.is_finite()),
        "nonfinite or incomplete full-vocabulary row",
    )?;
    let mut top1 = 0usize;
    for i in 1..row.len() {
        if row[i] > row[top1] {
            top1 = i;
        }
    }
    let raw: Vec<_> = row.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
    // This existing production diagnostic includes active QSA partial K/V,
    // unlike the session serialization subset. Every prior forward succeeded,
    // so its independent retained queue has already returned waited. This
    // diagnostic ALSO synchronizes the entire HIP device and drains copy DMA;
    // never interpret this instrumented schedule as async-overlap evidence.
    let state = bundle.state.canonical_digest_rows(gpu).map_err(err)?;
    Ok(Point {
        case: case.into(),
        phase: phase.into(),
        step,
        position: bundle.state.position,
        history_sha256: ids_sha(history),
        vocab: row.len(),
        logits_bytes: raw.len(),
        logits_sha256: sha(&raw),
        top1: top1 as u32,
        state,
    })
}
fn decode(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    logits: &GpuTensor,
    case: &str,
    phase: &str,
    steps: usize,
    history: &mut Vec<u32>,
    points: &mut Vec<Point>,
    reference: Option<&Report>,
) -> R<()> {
    for i in 0..steps {
        let token = points.last().ok_or("decode has no preceding logits")?.top1;
        bundle
            .forward_token(gpu, token, logits, None)
            .map_err(err)?;
        history.push(token);
        record_point(
            points,
            observe(bundle, gpu, logits, case, phase, i + 1, history)?,
            reference,
        )?;
    }
    Ok(())
}
fn suffix(tokens: &[u32], rows: usize, offset: usize) -> Vec<u32> {
    (0..rows)
        .map(|i| tokens[(i + offset) % tokens.len()])
        .collect()
}

fn known_cancel(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    logits: &GpuTensor,
    tokens: &[u32],
    initial: &Point,
    cache: bool,
) -> R<Value> {
    // One-shot failure BEFORE the first successful prefix kernel: the retained
    // body is reached only after that eager prefix. This exercises known-no-
    // independent-submit cancellation, never a real GPU hang or PM4 timeout.
    let prior = gpu.replay.shadow_body_route();
    gpu.replay
        .set_shadow_body_route(Some(rdna_compute::replay::ShadowBodyRoute::Hip));
    gpu.lab_arm_kernel_fault_after(0).map_err(err)?;
    let attempted = bundle.forward_token(gpu, tokens[0], logits, None);
    let fired = gpu.lab_disarm_kernel_fault();
    gpu.replay.set_shadow_body_route(prior);
    require(
        fired.injected && fired.successful_launches == 0,
        "fault did not fire before first prefix kernel",
    )?;
    let failure = attempted
        .err()
        .ok_or("faulted forward unexpectedly succeeded")?
        .to_string();
    require(
        failure.contains("lab: injected one-shot kernel launch failure"),
        "unrelated GPU error cannot pass known-cancel control",
    )?;
    // These actual bundle entrypoints reject an unretired/poisoned manager.
    let after_error = census(bundle, gpu, cache)?;
    bundle.reset(gpu).map_err(err)?;
    bundle
        .forward_chunk_final(gpu, tokens, logits, None)
        .map_err(err)?;
    let recovered = observe(
        bundle,
        gpu,
        logits,
        &initial.case,
        &initial.phase,
        initial.step,
        tokens,
    )?;
    require(
        &recovered == initial,
        "known-cancel reset/prefill recovery differs numerically",
    )?;
    Ok(json!({"injected":true,"successful_prefix_launches":0,
        "cache_idle_state_entry_recovered":true,"canonical_initial_state_and_full_logits_exact":true,
        "census_after_error":after_error,
        "scope":"known no-independent-submit eager-prefix cancellation only; not HIP wait/readback/table-upload/PM4 timeout injection"}))
}

struct Args {
    model: PathBuf,
    fixture: PathBuf,
    output: PathBuf,
    reference: Option<PathBuf>,
    cache: bool,
    execute: bool,
    known_cancel: bool,
    max_seq: usize,
    steps: usize,
}
fn args() -> R<Args> {
    let mut a = Args {
        model: PathBuf::new(),
        fixture: PathBuf::new(),
        output: PathBuf::new(),
        reference: None,
        cache: false,
        execute: false,
        known_cancel: false,
        max_seq: 16384,
        steps: 128,
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        match k.as_str() {
            "--execute" => a.execute = true,
            "--known-cancel" => a.known_cancel = true,
            "--model" => a.model = it.next().ok_or("missing model")?.into(),
            "--fixture" => a.fixture = it.next().ok_or("missing fixture")?.into(),
            "--output" => a.output = it.next().ok_or("missing output")?.into(),
            "--reference" => a.reference = Some(it.next().ok_or("missing reference")?.into()),
            "--cache" => {
                a.cache = match it.next().as_deref() {
                    Some("0") => false,
                    Some("1") => true,
                    _ => return Err("--cache requires 0 or 1".into()),
                }
            }
            "--max-seq" => a.max_seq = it.next().ok_or("missing max-seq")?.parse().map_err(err)?,
            "--decode-steps" => {
                a.steps = it
                    .next()
                    .ok_or("missing decode-steps")?
                    .parse()
                    .map_err(err)?
            }
            _ => return Err(format!("unknown argument {k}")),
        }
    }
    require(
        a.model.is_absolute() && a.fixture.is_absolute() && a.output.is_absolute(),
        "model/fixture/output must be absolute paths",
    )?;
    require(
        a.steps >= 64 && a.max_seq >= 8192,
        "qualification needs >=64 decode steps and >=8192 context",
    )?;
    require(
        !a.cache || a.reference.is_some(),
        "cache-on must compare an existing cache-off reference",
    )?;
    require(!a.output.exists(), "refusing to overwrite evidence")?;
    Ok(a)
}
fn run(a: &Args) -> R<Report> {
    let fixture_bytes = std::fs::read(&a.fixture).map_err(err)?;
    let fixture: Fixture = serde_json::from_slice(&fixture_bytes).map_err(err)?;
    require(!fixture.cases.is_empty(), "empty fixture")?;
    // Reject malformed/stale references before opening a GPU or loading weights.
    let reference: Option<Report> = a
        .reference
        .as_ref()
        .map(|path| -> R<Report> {
            serde_json::from_slice(&std::fs::read(path).map_err(err)?).map_err(err)
        })
        .transpose()?;
    if let Some(reference) = reference.as_ref() {
        check_structure(reference)?;
        require(
            !reference.actual_cache && a.cache,
            "reference must be cache-off for cache-on",
        )?;
        require(
            reference.fixture_sha256 == sha(&fixture_bytes)
                && reference.max_seq == a.max_seq
                && reference.decode_steps == a.steps,
            "reference fixture/context/decode schedule differs",
        )?;
    }
    let raw = a.model.to_str().ok_or("non-UTF8 model path")?;
    let mut source = ModelSource::from_path(raw)?;
    let source_identity = match &source {
        ModelSource::Hfq(h) => {
            let base = hipfire_arch_qwen4::expert_hotset::HotsetBase::from_hfq(h)?;
            PartitionSource {
                artifact_bytes: base.size(),
                artifact_index_sha256: base.index_sha256().into(),
                layers: hipfire_arch_qwen4::expert_slabs::TRUNK_LAYERS as u32,
                experts_per_layer: hipfire_arch_qwen4::expert_slabs::EXPERTS_PER_LAYER as u32,
            }
        }
        _ => return Err("requires reviewed HFQ artifact".into()),
    };
    if let Some(reference) = reference.as_ref() {
        require(
            reference.source_identity == source_identity,
            "actual HFQ source index/container differs before model load",
        )?;
    }
    let tokenizer = match &source {
        ModelSource::Hfq(h) => Tokenizer::from_hfq_metadata(&h.metadata_json).map_err(err)?,
        _ => return Err("requires reviewed HFQ artifact".into()),
    };
    // Resolve the same fixed hot budget as the product loader; never clamp it.
    require(
        hipfire_arch_qwen4::expert_hotset::HotsetBudget::from_env()?.select(u64::MAX)?
            == 20000 * (1u64 << 20),
        "requires exact matched 20000MiB hot budget",
    )?;
    require(
        hipfire_config::memory::ple_format() == hipfire_config::memory::PleFormatRequest::Q4,
        "requires reviewed q4 PLE",
    )?;
    require(
        hipfire_config::developer_var("HIPFIRE_OFFLOAD_EXEC")
            .ok()
            .as_deref()
            == Some("pcie"),
        "requires explicit PCIe execution",
    )?;
    hipfire_loader::prepare_host_memory_for(&a.model);
    let mut gpu = Gpu::init().map_err(err)?;
    require(
        gpu.arch == "gfx1201",
        "qualification requires exact gfx1201",
    )?;
    hipfire_loader::admission::resolve_qwen4_ple_rows(&mut source, raw, &gpu.arch)?;
    let backend = Qwen4KvBackend::automatic(&gpu);
    require(backend.name() == "vmm", "requires actual automatic VMM")?;
    let shared = Qwen4RankShared {
        path: raw.into(),
        max_seq: a.max_seq,
        sequence: None,
        backend,
        vmm_granularity: gpu.vmm_recommended_granularity().map_err(err)?,
        kv_mode: "fp8".into(),
        state_quant: Some("q8".into()),
        spec: SpecLoadCfg {
            mtp: Some(false),
            ngram_draft: Some(false),
            ..SpecLoadCfg::default()
        },
        native_mtp: false,
        max_k: 0,
        batch: Default::default(),
    };
    let loaded = load_qwen4_rank(&mut gpu, None, &shared, None, None, source)?;
    let native_mtp = loaded.native_mtp;
    let max_seq = loaded.max_seq;
    let mut bundle = loaded.bundle;
    let logits = gpu
        .zeros(&[bundle.config.vocab_size], DType::F32)
        .map_err(err)?;
    let result = (|| -> R<Report> {
        require(
            !native_mtp,
            "product loader attached native MTP in AR-only oracle",
        )?;
        require(
            bundle.expert_cache_enabled() == a.cache,
            "product loader did not admit requested actual cache arm",
        )?;
        require(
            bundle.state_format().gdn.name() == "q8"
                && format!("{:?}", bundle.state_format().qsa) == "Fp8",
            "actual state must remain plain Q8 GDN and native FP8 QSA",
        )?;
        let lease = bundle
            .weights
            .expert_tiers()
            .ok_or("missing product hot-set tier owners")?;
        let physical_plan_sha256 = lease
            .logical_digest()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        let (semantic_partition_sha256, original_resident_experts) =
            actual_semantic_partition(lease, &source_identity)?;
        if let Some(reference) = reference.as_ref() {
            require(
                reference.semantic_partition_sha256 == semantic_partition_sha256,
                "actual original expert residency/codecs/source geometry differ",
            )?;
        }
        let hotset_sha256 = lease
            .hotset_sidecar_sha256()
            .ok_or("missing actual sidecar pin")?
            .to_string();
        let donor_bytes = lease
            .expert_cache_ranges()
            .map_or(0, |r| r.donor_device.1 as u64);
        require(
            !a.cache || donor_bytes > 0,
            "cache arm has no actual donor pool",
        )?;
        let mut points = Vec::new();
        let mut cells = Vec::new();
        let mut last_initial = None;
        let mut last_tokens = Vec::new();
        for case in &fixture.cases {
            let text = std::fs::read_to_string(&case.prompt_file).map_err(err)?;
            require(case.prompt_rows > 0, "base prompt rows must be nonzero")?;
            let tokens: Vec<_> = tokenizer
                .encode(&text)
                .into_iter()
                .take(case.prompt_rows)
                .collect();
            require(
                !tokens.is_empty()
                    && tokens.len() + case.short_rows + case.staged_rows + 2 * a.steps + 32
                        < max_seq,
                "fixture exceeds admitted context",
            )?;
            require(case.short_rows > 0, "short suffix must be nonempty")?;
            bundle.reset(&mut gpu).map_err(err)?;
            bundle
                .forward_chunk_final(&mut gpu, &tokens, &logits, None)
                .map_err(err)?;
            let mut history = tokens.clone();
            let initial = observe(
                &mut bundle,
                &mut gpu,
                &logits,
                &case.id,
                "base_prefill",
                0,
                &history,
            )?;
            record_point(&mut points, initial.clone(), reference.as_ref())?;
            decode(
                &mut bundle,
                &mut gpu,
                &logits,
                &case.id,
                "decode_before_short",
                a.steps,
                &mut history,
                &mut points,
                reference.as_ref(),
            )?;
            let before_short = census(&bundle, &gpu, a.cache)?;
            if a.cache {
                require(
                    before_short["dirty"] == true,
                    "short-prefill prerequisite donor dirty was not observed",
                )?;
                let threshold = before_short["stage_min_rows"]
                    .as_u64()
                    .ok_or("missing stage threshold")? as usize;
                require(
                    case.short_rows < threshold && case.staged_rows >= threshold,
                    "fixture does not cross actual staging boundary",
                )?;
            }
            let short = suffix(&tokens, case.short_rows, 7);
            bundle
                .forward_chunk_final(&mut gpu, &short, &logits, None)
                .map_err(err)?;
            history.extend(short);
            record_point(
                &mut points,
                observe(
                    &mut bundle,
                    &mut gpu,
                    &logits,
                    &case.id,
                    "dirty_short_prefill",
                    0,
                    &history,
                )?,
                reference.as_ref(),
            )?;
            let after_short = census(&bundle, &gpu, a.cache)?;
            if a.cache {
                require(
                    delta(&before_short, &after_short, "mapped_prefill_calls")? > 0
                        && delta(&before_short, &after_short, "mapped_prefill_rows")?
                            == case.short_rows as u64
                        && delta(&before_short, &after_short, "prefill_restores")? == 0
                        && delta(&before_short, &after_short, "restore_bytes")? == 0,
                    "short prefill did not use mapped view without bulk restoration",
                )?;
            }
            decode(
                &mut bundle,
                &mut gpu,
                &logits,
                &case.id,
                "decode_before_staged",
                a.steps,
                &mut history,
                &mut points,
                reference.as_ref(),
            )?;
            let before_long = census(&bundle, &gpu, a.cache)?;
            if a.cache {
                require(
                    before_long["dirty"] == true,
                    "staged restore prerequisite not dirty",
                )?;
            }
            let long = suffix(&tokens, case.staged_rows, 19);
            bundle
                .forward_chunk_final(&mut gpu, &long, &logits, None)
                .map_err(err)?;
            history.extend(long);
            record_point(
                &mut points,
                observe(
                    &mut bundle,
                    &mut gpu,
                    &logits,
                    &case.id,
                    "staged_long_prefill",
                    0,
                    &history,
                )?,
                reference.as_ref(),
            )?;
            let after_long = census(&bundle, &gpu, a.cache)?;
            if a.cache {
                require(
                    delta(&before_long, &after_long, "prefill_restores")? == 1
                        && delta(&before_long, &after_long, "restore_bytes")? == donor_bytes,
                    "long prefill did not restore the exact donor bytes once",
                )?;
            }
            decode(
                &mut bundle,
                &mut gpu,
                &logits,
                &case.id,
                "decode_after_restore",
                32,
                &mut history,
                &mut points,
                reference.as_ref(),
            )?;
            cells.push(json!({"case":case.id,"prompt_file_sha256":sha(text.as_bytes()),
                "base_prompt_ids_sha256":ids_sha(&tokens),"before_short":before_short,"after_short":after_short,
                "before_long":before_long,"after_long":after_long}));
            last_initial = Some(initial);
            last_tokens = tokens;
        }
        let end = census(&bundle, &gpu, a.cache)?;
        if a.cache {
            for k in [
                "tokens",
                "cache_hits",
                "fills_started",
                "fills_completed",
                "fill_bytes",
            ] {
                require(
                    count(&end, k)? > 0,
                    format!("actual cache execution lacked {k}"),
                )?;
            }
        }
        let fault = if a.known_cancel {
            known_cancel(
                &mut bundle,
                &mut gpu,
                &logits,
                &last_tokens,
                last_initial.as_ref().ok_or("no initial point")?,
                a.cache,
            )?
        } else {
            json!({"observed":false,"reason":"not requested"})
        };
        let report = Report {
            schema: "hipfire-fn-ar-lru-numerical-v2".into(),
            requested_cache: a.cache,
            actual_cache: bundle.expert_cache_enabled(),
            native_mtp,
            max_seq,
            decode_steps: a.steps,
            fixture_sha256: sha(&fixture_bytes),
            hotset_sha256,
            physical_plan_sha256,
            semantic_partition_sha256,
            source_identity: source_identity.clone(),
            state_format: format!("{:?}", bundle.state_format()),
            kv_backend: bundle.state.qsa_backend().name().into(),
            points,
            coverage: json!({"cases":cells,"census_end":end,"donor_bytes":donor_bytes,"known_cancel":fault}),
            identity: json!({"model":a.model,"model_bytes":std::fs::metadata(&a.model).map_err(err)?.len(),
                "loader":"production load_qwen4_rank","native_mtp":false,"max_k":0,
                "hot_budget_mib":20000,"original_resident_experts":original_resident_experts,
                "partition_comparison":"original Resident entry tier+qtype pair+projection lengths bound to actual HFQ size/index; physical plan recorded separately",
                "artifact_payload_sha256":"NOT_REHASHED_BY_THIS_TOOL; supervisor frozen artifact seal required",
                "logits":"full-vocabulary F32 LE bits",
                "state":"canonical_digest_rows active target bytes and semantic marks",
                "instrumentation":"INTRUSIVE: canonical_digest_rows performs device_synchronize and drains pending copy DMA",
                "async_overlap_proof":false,
                "short_prefill_counter_scope":"no additional manager bulk restore/copy; not pending-DMA overlap",
                "performance_proof":false,
                "served_fast_head_interchangeability":"NOT_CLAIMED",
                "raw_fp8_vmm_startup_marker":"RETAIN_IN_SUPERVISOR_LOG",
                "backend_observation":"RETAIN_ACTUAL_ROUTE_PROOF_IN_SUPERVISOR_LOG"}),
            performance_claim: false,
        };
        check_structure(&report)?;
        if let Some(reference) = reference.as_ref() {
            compare(reference, &report)?;
        }
        Ok(report)
    })();
    match result {
        Ok(report) => {
            // Close readers/copy streams before freeing the caller's output.
            if let Err(error) = bundle.free_gpu(&mut gpu) {
                std::mem::forget(logits);
                std::mem::forget(gpu);
                return Err(format!("bundle shutdown failed; owners retained: {error}"));
            }
            gpu.free_tensor(logits).map_err(err)?;
            gpu.drain_pool();
            Ok(report)
        }
        Err(primary) => {
            // Do not turn a plain HIP sync into independent-queue retirement.
            // Preserve every owner; the process exits and no load is retried.
            std::mem::forget(logits);
            std::mem::forget(bundle);
            std::mem::forget(gpu);
            Err(primary)
        }
    }
}
fn main() {
    let result = (|| -> R<()> {
        let a = args()?;
        if !a.execute {
            println!(
                "{}",
                json!({"status":"PLAN_ONLY","gpu_loads":0,
                "action":"one product AR-only load; cache-off reference then separate cache-on process",
                "requires_external_gpu_locks":true,"requires_lab_census_getters":true})
            );
            return Ok(());
        }
        let report = run(&a)?;
        std::fs::write(&a.output, serde_json::to_vec_pretty(&report).map_err(err)?).map_err(err)?;
        Ok(())
    })();
    if let Err(error) = result {
        eprintln!("AR numerical gate FAILED: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn source_fixture() -> PartitionSource {
        PartitionSource {
            artifact_bytes: 1000,
            artifact_index_sha256: sha(b"real-index-fixture"),
            layers: 1,
            experts_per_layer: 2,
        }
    }
    fn expert_fixture() -> Vec<SemanticExpert> {
        (0..2)
            .map(|expert| SemanticExpert {
                layer: 0,
                expert,
                original_device: expert == 0,
                gate_up_qtype: 44,
                down_qtype: 53,
                gate_up_bytes: 1740800,
                down_bytes: 870400,
            })
            .collect()
    }
    fn fixture(cache: bool) -> Report {
        Report {
            schema: "hipfire-fn-ar-lru-numerical-v2".into(),
            requested_cache: cache,
            actual_cache: cache,
            native_mtp: false,
            max_seq: 16384,
            decode_steps: 128,
            fixture_sha256: "f".repeat(64),
            hotset_sha256: "h".repeat(64),
            physical_plan_sha256: sha(if cache {
                b"mirror-physical-plan"
            } else {
                b"plain-physical-plan"
            }),
            semantic_partition_sha256: semantic_partition_digest(
                &source_fixture(),
                &expert_fixture(),
            )
            .unwrap(),
            source_identity: source_fixture(),
            state_format: "FP8/plainQ8".into(),
            kv_backend: "vmm".into(),
            points: vec![Point {
                case: "x".into(),
                phase: "decode".into(),
                step: 1,
                position: 7,
                history_sha256: "i".repeat(64),
                vocab: 8,
                logits_bytes: 32,
                logits_sha256: sha(&[0u8; 32]),
                top1: 2,
                state: vec![
                    ("gdn[L0].recurrent".into(), sha(&[0u8; 4])),
                    ("qsa[L3].full_keys".into(), sha(&[1u8; 4])),
                ],
            }],
            coverage: Value::Null,
            identity: Value::Null,
            performance_claim: false,
        }
    }
    #[test]
    fn full_vocabulary_reference_bit_flip_fails() {
        let off = fixture(false);
        let mut on = fixture(true);
        assert!(compare(&off, &on).is_ok());
        let mut bits = [0u8; 32];
        bits[0] ^= 1;
        on.points[0].logits_sha256 = sha(&bits);
        assert!(compare(&off, &on).is_err());
    }
    #[test]
    fn truncated_reference_rows_fail() {
        let mut off = fixture(false);
        let on = fixture(true);
        assert!(compare(&off, &on).is_ok());
        off.points.clear();
        assert!(compare(&off, &on).is_err());
    }
    #[test]
    fn missing_target_state_part_fails() {
        let off = fixture(false);
        let mut on = fixture(true);
        assert!(compare(&off, &on).is_ok());
        on.points[0].state.pop();
        assert!(compare(&off, &on).is_err());
    }
    #[test]
    fn disabled_cache_cannot_pass_on_arm() {
        let off = fixture(false);
        let mut on = fixture(true);
        assert!(compare(&off, &on).is_ok());
        on.actual_cache = false;
        assert!(compare(&off, &on).is_err());
    }
    #[test]
    fn mirror_physical_layout_can_differ_without_relaxing_original_partition() {
        let off = fixture(false);
        let on = fixture(true);
        assert_ne!(off.physical_plan_sha256, on.physical_plan_sha256);
        assert_eq!(off.semantic_partition_sha256, on.semantic_partition_sha256);
        assert!(compare(&off, &on).is_ok());
    }
    #[test]
    fn genuine_original_resident_set_change_is_rejected() {
        let off = fixture(false);
        let mut on = fixture(true);
        assert!(compare(&off, &on).is_ok());
        let mut rows = expert_fixture();
        rows[0].original_device = false;
        rows[1].original_device = true; // same resident count, different expert IDs
        on.semantic_partition_sha256 = semantic_partition_digest(&source_fixture(), &rows).unwrap();
        assert!(compare(&off, &on).is_err());
    }
    #[test]
    fn source_geometry_and_projection_format_changes_are_rejected() {
        let off = fixture(false);
        let on = fixture(true);
        assert!(compare(&off, &on).is_ok());
        let mut rows = expert_fixture();
        rows[0].gate_up_qtype = 131;
        let mut changed = on.clone();
        changed.semantic_partition_sha256 =
            semantic_partition_digest(&source_fixture(), &rows).unwrap();
        assert!(compare(&off, &changed).is_err());
        rows = expert_fixture();
        rows[0].down_bytes += 256;
        changed = on.clone();
        changed.semantic_partition_sha256 =
            semantic_partition_digest(&source_fixture(), &rows).unwrap();
        assert!(compare(&off, &changed).is_err());
        changed = on.clone();
        changed.source_identity.artifact_index_sha256 = sha(b"different-source-index");
        changed.semantic_partition_sha256 =
            semantic_partition_digest(&changed.source_identity, &expert_fixture()).unwrap();
        assert!(compare(&off, &changed).is_err());
        changed = on.clone();
        changed.source_identity.artifact_bytes += 1;
        changed.semantic_partition_sha256 =
            semantic_partition_digest(&changed.source_identity, &expert_fixture()).unwrap();
        assert!(compare(&off, &changed).is_err());
    }
    #[test]
    fn missing_or_duplicate_partition_rows_cannot_form_a_fingerprint() {
        let mut rows = expert_fixture();
        rows.pop();
        assert!(semantic_partition_digest(&source_fixture(), &rows).is_err());
        rows = expert_fixture();
        rows[1].expert = 0;
        assert!(semantic_partition_digest(&source_fixture(), &rows).is_err());
    }
}
