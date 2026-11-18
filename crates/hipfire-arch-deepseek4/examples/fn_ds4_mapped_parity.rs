// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU gate of the DeepSeek V4 pinned-host canonical expert owner (expert cache slice S6b).
//!
//! ```text
//! fn_ds4_mapped_parity <model.hfq> <layer> <out.json>
//!     [--expert E]          expert of part (a) (default 7)
//!     [--trials N]          routes of part (b) (default 8)
//!     [--steps N]           greedy tokens of part (c) (default 16)
//!     [--prompt TEXT]       part (c) prompt (default "The capital of France is")
//!     [--source mapped|resident]
//!                           part (c) loader (default mapped). `resident` runs part (c) only:
//!                           it is the reference run a mapped run is compared against.
//!     [--mtp]               include the MTP head's experts in part (c)'s source
//!     [--skip-full-load]    run parts (a) and (b) only
//!     [--compare OTHER.json] compare part (c)'s token ids and per-step logits hashes with the
//!                           `parts.c_full_load.decode` of another run's JSON
//! ```
//!
//! Parts (`--source mapped`):
//!
//! * (a) `PreadSource::fetch_into_batch` H2D byte identity: the three file extents of one real
//!   expert (`Ds4ExpertSource::file_copies`) land in a device allocation at the slot offsets and
//!   read back byte-identical to an independent positional read of the file and to the expert's
//!   span in the pinned blob. A sparse file one byte shorter than every extent refuses before any
//!   copy (the device allocation stays at its sentinel), and an out-of-bounds destination refuses
//!   before any read.
//! * (b) one-layer parity: layer `<layer>`'s routed experts loaded as resident device blobs and
//!   as pinned host-mapped blobs run the production selected-experts dispatch over the same
//!   routes and inputs; the outputs must be bit-identical (and non-zero, and different across
//!   different routes, so identity is not vacuous).
//! * (c) full-model load with the source, `host_charge()` against the pinned bytes the registry
//!   and `/proc/meminfo` show, then a short greedy decode whose token ids and per-step logits
//!   hashes go to the JSON.
//!
//! The JSON is rewritten after every part; the exit status is non-zero unless every part that
//! ran passed. Scratch files go under `TMPDIR` (`std::env::temp_dir()`).

use hipfire_arch_deepseek4::expert_source::{
    check_host_ram, ExpertSourceMode, HostCharge, HOST_TAIL_PAD_BYTES,
};
use hipfire_arch_deepseek4::forward::decode_step;
use hipfire_arch_deepseek4::lab_mapped::{self, ProbeLayer};
use hipfire_arch_deepseek4::DeepseekV4;
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::expert_cache::pread::{FetchReq, PreadSource};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::Gpu;
use serde_json::{json, Value};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

const SENTINEL: u8 = 0xA5;

struct Args {
    model: PathBuf,
    layer: usize,
    out: PathBuf,
    expert: u32,
    trials: usize,
    steps: usize,
    prompt: String,
    mapped: bool,
    mtp: bool,
    skip_full_load: bool,
    compare: Option<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let usage = "usage: fn_ds4_mapped_parity <model.hfq> <layer> <out.json> [--expert E] \
                 [--trials N] [--steps N] [--prompt TEXT] [--source mapped|resident] [--mtp] \
                 [--skip-full-load] [--compare OTHER.json]";
    let mut it = std::env::args().skip(1);
    let mut next = |what: &str| it.next().ok_or_else(|| format!("missing {what}\n{usage}"));
    let model = PathBuf::from(next("<model.hfq>")?);
    let layer = next("<layer>")?
        .parse()
        .map_err(|e| format!("<layer>: {e}"))?;
    let out = PathBuf::from(next("<out.json>")?);
    let mut args = Args {
        model,
        layer,
        out,
        expert: 7,
        trials: 8,
        steps: 16,
        prompt: "The capital of France is".to_string(),
        mapped: true,
        mtp: false,
        skip_full_load: false,
        compare: None,
    };
    while let Some(flag) = it.next() {
        let value = |it: &mut std::iter::Skip<std::env::Args>| {
            it.next()
                .ok_or_else(|| format!("{flag} needs a value\n{usage}"))
        };
        match flag.as_str() {
            "--expert" => {
                args.expert = value(&mut it)?
                    .parse()
                    .map_err(|e| format!("{flag}: {e}"))?
            }
            "--trials" => {
                args.trials = value(&mut it)?
                    .parse()
                    .map_err(|e| format!("{flag}: {e}"))?
            }
            "--steps" => {
                args.steps = value(&mut it)?
                    .parse()
                    .map_err(|e| format!("{flag}: {e}"))?
            }
            "--prompt" => args.prompt = value(&mut it)?,
            "--source" => {
                args.mapped = match value(&mut it)?.as_str() {
                    "mapped" => true,
                    "resident" => false,
                    other => return Err(format!("--source {other}: expected mapped|resident")),
                }
            }
            "--mtp" => args.mtp = true,
            "--skip-full-load" => args.skip_full_load = true,
            "--compare" => args.compare = Some(PathBuf::from(value(&mut it)?)),
            _ => return Err(format!("unknown flag {flag}\n{usage}")),
        }
    }
    Ok(args)
}

fn write_report(path: &Path, report: &Value) -> Result<(), String> {
    let text = serde_json::to_string_pretty(report).map_err(|e| e.to_string())?;
    std::fs::write(path, text).map_err(|e| format!("write {}: {e}", path.display()))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn fnv64(words: impl IntoIterator<Item = u32>) -> u64 {
    words
        .into_iter()
        .fold(0xcbf2_9ce4_8422_2325u64, |hash, word| {
            word.to_le_bytes().iter().fold(hash, |hash, &byte| {
                (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
            })
        })
}

fn first_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    a.iter()
        .zip(b)
        .position(|(x, y)| x != y)
        .or_else(|| (a.len() != b.len()).then(|| a.len().min(b.len())))
}

fn charge_json(charge: &HostCharge) -> Value {
    json!({
        "blob_bytes": charge.blob_bytes,
        "tail_pad_bytes": charge.tail_pad_bytes,
        "allocations": charge.allocations,
        "total": charge.total(),
    })
}

fn meminfo() -> Value {
    let mut fields = serde_json::Map::new();
    if let Ok(text) = std::fs::read_to_string("/proc/meminfo") {
        for line in text.lines() {
            let mut parts = line.split_whitespace();
            let (Some(name), Some(value)) = (parts.next(), parts.next()) else {
                continue;
            };
            let name = name.trim_end_matches(':');
            if matches!(
                name,
                "MemTotal"
                    | "MemFree"
                    | "MemAvailable"
                    | "Unevictable"
                    | "Mlocked"
                    | "AnonPages"
                    | "Shmem"
                    | "Cached"
            ) {
                if let Ok(kib) = value.parse::<u64>() {
                    fields.insert(name.to_string(), json!(kib * 1024));
                }
            }
        }
    }
    Value::Object(fields)
}

/// Part (a): `fetch_into_batch` of one real expert, plus the short-file and out-of-bounds
/// refusals.
fn part_a(
    gpu: &mut Gpu,
    hfq: &HfqFile,
    source: &hipfire_arch_deepseek4::expert_source::Ds4ExpertSource,
    args: &Args,
    mapped: &ProbeLayer,
) -> Result<Value, String> {
    let key = source.key(args.layer as u32, args.expert);
    let copies = source.file_copies(&key).map_err(|e| e.to_string())?;
    let bundle = source.expert_bytes().bundle() as usize;
    let max_extent = copies
        .iter()
        .map(|c| c.extent.len as usize)
        .max()
        .unwrap_or(0);

    // The expected slot image, from an independent positional read of the file.
    let file = std::fs::File::open(hfq.path()).map_err(|e| format!("open model: {e}"))?;
    let mut expected = vec![0u8; bundle];
    for copy in &copies {
        let at = copy.slot_offset as usize;
        file.read_exact_at(
            &mut expected[at..at + copy.extent.len as usize],
            copy.extent.offset,
        )
        .map_err(|e| format!("independent read of {:?}: {e}", copy.part))?;
    }

    let dst = gpu
        .hip
        .malloc(bundle)
        .map_err(|e| format!("malloc slot: {e:?}"))?;
    let result = (|| -> Result<Value, String> {
        let reqs: Vec<FetchReq<'_>> = copies
            .iter()
            .map(|copy| FetchReq {
                key,
                file_offset: copy.extent.offset,
                len: copy.extent.len as usize,
                dst: &dst,
                dst_offset: copy.slot_offset as usize,
            })
            .collect();
        let mut pread = PreadSource::open(hfq.path(), max_extent).map_err(|e| e.to_string())?;
        gpu.hip
            .memset(&dst, i32::from(SENTINEL), bundle)
            .map_err(|e| format!("memset sentinel: {e:?}"))?;
        pread
            .fetch_into_batch(&reqs, &gpu.hip)
            .map_err(|e| format!("fetch_into_batch: {e}"))?;
        let mut back = vec![0u8; bundle];
        gpu.hip
            .memcpy_dtoh(&mut back, &dst)
            .map_err(|e| format!("read back: {e:?}"))?;
        let byte_identical = back == expected;

        // The same expert's span inside the pinned blobs of the mapped layer load.
        let blob_identical = match mapped.group() {
            Some(group) => {
                let bytes = source.expert_bytes();
                let (gate_up, down) = (bytes.gate_up() as usize, bytes.down as usize);
                let e = args.expert as usize;
                fn host_slice<'a>(
                    gpu: &Gpu,
                    tensor: &'a Option<rdna_compute::GpuTensor>,
                ) -> Option<&'a [u8]> {
                    tensor.as_ref().and_then(|t| gpu.host_bytes(t))
                }
                let (gu_blob, dn_blob) = (
                    host_slice(gpu, &mapped.layer().expert_gate_up_blob),
                    host_slice(gpu, &mapped.layer().expert_w2_blob),
                );
                let _ = group;
                match (gu_blob, dn_blob) {
                    (Some(gu), Some(dn)) => Some(
                        gu[e * gate_up..(e + 1) * gate_up] == expected[..gate_up]
                            && dn[e * down..(e + 1) * down] == expected[gate_up..gate_up + down],
                    ),
                    _ => None,
                }
            }
            None => None,
        };

        // A sparse file one byte shorter than the shortest extent end: every read is short.
        let shortest_end = copies
            .iter()
            .map(|c| c.extent.offset + c.extent.len)
            .min()
            .unwrap_or(1);
        let short_path = std::env::temp_dir().join(format!(
            "fn_ds4_mapped_parity_short_{}.bin",
            std::process::id()
        ));
        let short = std::fs::File::create(&short_path)
            .and_then(|f| f.set_len(shortest_end - 1))
            .map_err(|e| format!("create short file {}: {e}", short_path.display()))?;
        let _ = short;
        let mut short_source =
            PreadSource::open(&short_path, max_extent).map_err(|e| e.to_string())?;
        gpu.hip
            .memset(&dst, i32::from(SENTINEL), bundle)
            .map_err(|e| format!("memset sentinel: {e:?}"))?;
        let short_error = short_source.fetch_into_batch(&reqs, &gpu.hip).err();
        let mut untouched = vec![0u8; bundle];
        gpu.hip
            .memcpy_dtoh(&mut untouched, &dst)
            .map_err(|e| format!("read back after refusal: {e:?}"))?;
        let _ = std::fs::remove_file(&short_path);

        // A destination range past the allocation refuses before any read.
        let oob = [FetchReq {
            key,
            file_offset: copies[0].extent.offset,
            len: copies[0].extent.len as usize,
            dst: &dst,
            dst_offset: bundle - 1,
        }];
        let oob_error = pread.fetch_into_batch(&oob, &gpu.hip).err();

        let passed = byte_identical
            && blob_identical != Some(false)
            && short_error.is_some()
            && untouched.iter().all(|&b| b == SENTINEL)
            && oob_error.is_some();
        Ok(json!({
            "pass": passed,
            "key": { "group": key.group_id, "expert": key.expert_id },
            "extents": copies.iter().map(|c| json!({
                "part": format!("{:?}", c.part),
                "file_offset": c.extent.offset,
                "len": c.extent.len,
                "slot_offset": c.slot_offset,
            })).collect::<Vec<_>>(),
            "bundle_bytes": bundle,
            "h2d_byte_identical": byte_identical,
            "first_difference": first_difference(&back, &expected),
            "pinned_blob_span_identical": blob_identical,
            "image_fnv64": format!("{:016x}", fnv64(expected.chunks(4).map(|c| {
                let mut w = [0u8; 4];
                w[..c.len()].copy_from_slice(c);
                u32::from_le_bytes(w)
            }))),
            "short_file": {
                "len": shortest_end - 1,
                "refused": short_error.is_some(),
                "error": short_error.as_ref().map(|e| e.to_string()),
                "device_untouched": untouched.iter().all(|&b| b == SENTINEL),
            },
            "out_of_bounds_destination": {
                "refused": oob_error.is_some(),
                "error": oob_error.as_ref().map(|e| e.to_string()),
            },
        }))
    })();
    let _ = gpu.hip.free(dst);
    result
}

/// Deterministic route `t`: `k` distinct expert ids (trial 0 pins the first and last expert),
/// positive route weights, and a rotated-space input.
fn route(t: usize, n: usize, k: usize, hidden: usize) -> (Vec<i32>, Vec<f32>, Vec<f32>) {
    let mut state = 0x9e37_79b9_7f4a_7c15u64 ^ ((t as u64 + 1) * 0xbf58_476d_1ce4_e5b9);
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut ids: Vec<i32> = Vec::with_capacity(k);
    if t == 0 {
        ids.extend([0, n as i32 - 1]);
    }
    while ids.len() < k {
        let id = (next() % n as u64) as i32;
        if !ids.contains(&id) {
            ids.push(id);
        }
    }
    let weights = (0..k).map(|i| (i + 1) as f32 / 21.0).collect();
    let x = (0..hidden)
        .map(|_| (next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0)
        .collect();
    (ids, weights, x)
}

/// Part (b): the resident and mapped loads of one layer run the same routes bit-identically.
fn part_b(
    gpu: &mut Gpu,
    cfg: &hipfire_arch_deepseek4::DeepseekV4Config,
    resident: &ProbeLayer,
    mapped: &ProbeLayer,
    args: &Args,
) -> Result<Value, String> {
    let (n, k, hidden) = (
        cfg.n_routed_experts,
        cfg.num_experts_per_tok,
        cfg.hidden_size,
    );
    let mut trials = Vec::new();
    let mut outputs: Vec<Vec<f32>> = Vec::new();
    let mut all_identical = true;
    for t in 0..args.trials {
        let (ids, weights, x) = route(t, n, k, hidden);
        let a = lab_mapped::run_selected(cfg, gpu, resident.layer(), &x, &ids, &weights)?;
        let b = lab_mapped::run_selected(cfg, gpu, mapped.layer(), &x, &ids, &weights)?;
        let bits = |v: &[f32]| {
            v.iter()
                .flat_map(|f| f.to_bits().to_le_bytes())
                .collect::<Vec<u8>>()
        };
        let (ab, bb) = (bits(&a), bits(&b));
        let identical = ab == bb;
        let finite_nonzero = a.iter().all(|v| v.is_finite()) && a.iter().any(|&v| v != 0.0);
        all_identical &= identical && finite_nonzero;
        trials.push(json!({
            "trial": t,
            "ids": ids,
            "bit_identical": identical,
            "first_difference_byte": first_difference(&ab, &bb),
            "output_finite_nonzero": finite_nonzero,
            "output_fnv64": format!("{:016x}", fnv64(a.iter().map(|f| f.to_bits()))),
        }));
        outputs.push(a);
    }
    // Negative control: different routes give different outputs, so identity is not the
    // output being constant.
    let distinct = outputs.windows(2).all(|pair| pair[0] != pair[1]);
    Ok(json!({
        "pass": all_identical && distinct && !trials.is_empty(),
        "layer": args.layer,
        "all_bit_identical": all_identical,
        "routes_give_distinct_outputs": distinct,
        "trials": trials,
    }))
}

/// Part (c): full-model load (mapped or resident), charge reconciliation, short greedy decode.
fn part_c(
    gpu: &mut Gpu,
    hfq: &mut HfqFile,
    cfg: &mut hipfire_arch_deepseek4::DeepseekV4Config,
    args: &Args,
) -> Result<Value, String> {
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|e| format!("tokenizer from HFQ metadata: {e:?}"))?;
    let prompt = tokenizer.encode(&args.prompt);
    if prompt.is_empty() {
        return Err("the prompt tokenizes to nothing".into());
    }
    cfg.load_dspark = false;
    cfg.load_mtp = args.mtp;
    let mode = if args.mapped {
        ExpertSourceMode::Mapped
    } else {
        ExpertSourceMode::Resident
    };

    let before = meminfo();
    // The measure the loader's admission reads (`pinnable_host_ram`), sampled just before it.
    let host_ram = hipfire_runtime::offload::pinnable_host_ram(Some(gpu));
    let host_ram = json!({
        "mem_available_minus_cma": host_ram.mem_available,
        "ttm_pool": host_ram.ttm_pool,
    });
    let started = std::time::Instant::now();
    let weights = match DeepseekV4::load_weights_expert_source(hfq, cfg, gpu, mode) {
        Ok(weights) => weights,
        Err(error) => {
            return Ok(json!({
                "pass": false,
                "source": format!("{mode:?}"),
                "load_error": error,
                "meminfo_before": before,
                "pinnable_host_ram_before": host_ram,
            }));
        }
    };
    let load_seconds = started.elapsed().as_secs_f64();
    let after = meminfo();

    let mut layers: Vec<&hipfire_arch_deepseek4::deepseek4::DeepseekV4LayerWeights> =
        weights.layers.iter().collect();
    layers.extend(weights.mtp_layer.as_ref());
    let host_blob = |t: &Option<rdna_compute::GpuTensor>| {
        t.as_ref()
            .and_then(|t| gpu.host_bytes(t))
            .map(|b| b.len() as u64)
    };
    let mut registry_blob_bytes = 0u64;
    let mut registry_allocations = 0usize;
    for layer in &layers {
        for blob in [&layer.expert_gate_up_blob, &layer.expert_w2_blob] {
            if let Some(len) = host_blob(blob) {
                registry_blob_bytes += len;
                registry_allocations += 1;
            }
        }
    }
    let registry_bytes = registry_blob_bytes + HOST_TAIL_PAD_BYTES * registry_allocations as u64;
    let mut charge_report = Value::Null;
    let mut reconciled = !args.mapped;
    if let Some(owner) = weights.mapped_expert_owner() {
        let charge = owner.source().host_charge();
        reconciled = charge.total() == registry_bytes
            && charge.allocations == registry_allocations
            && owner.observed_charge() == charge;
        let delta = |name: &str| -> Option<i64> {
            Some(before.get(name)?.as_u64()? as i64 - after.get(name)?.as_u64()? as i64)
        };
        charge_report = json!({
            "host_charge": charge_json(&charge),
            "owner_observed_charge": charge_json(&owner.observed_charge()),
            "registry_blob_bytes": registry_blob_bytes,
            "registry_allocations": registry_allocations,
            "registry_bytes_with_tail_pads": registry_bytes,
            "gpu_host_mapped_count": gpu.host_mapped_count(),
            "owner_id": owner.owner_id(),
            "owner_generation": owner.owner_generation(),
            "groups": owner.groups().len(),
            "meminfo_mem_available_drop": delta("MemAvailable"),
            "meminfo_unevictable_growth": delta("Unevictable").map(|d| -d),
            "meminfo_mlocked_growth": delta("Mlocked").map(|d| -d),
            "reconciled": reconciled,
        });
    }

    let mut state = DeepseekV4::new_state(gpu, cfg)?;
    let mut steps = Vec::new();
    let mut generated = Vec::new();
    let mut token = prompt[0];
    let total = prompt.len() + args.steps;
    for pos in 0..total {
        let logits = decode_step(cfg, &weights, &mut state, gpu, token, pos as u32)?;
        let argmax = logits
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |best, (i, &v)| {
                if v > best.1 {
                    (i, v)
                } else {
                    best
                }
            })
            .0 as u32;
        let mut top: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        top.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        steps.push(json!({
            "pos": pos,
            "input_token": token,
            "argmax": argmax,
            "logits_fnv64": format!("{:016x}", fnv64(logits.iter().map(|f| f.to_bits()))),
            "top8": top.iter().take(8).map(|&(i, v)| json!([i, v.to_bits()])).collect::<Vec<_>>(),
        }));
        token = if pos + 1 < prompt.len() {
            prompt[pos + 1]
        } else {
            generated.push(argmax);
            argmax
        };
    }
    state.free_gpu(gpu);
    weights.free_gpu(gpu);
    let host_mapped_after_free = gpu.host_mapped_count();

    Ok(json!({
        "pass": reconciled && host_mapped_after_free == 0,
        "source": format!("{mode:?}"),
        "load_seconds": load_seconds,
        "charge": charge_report,
        "host_mapped_after_free": host_mapped_after_free,
        "meminfo_before": before,
        "pinnable_host_ram_before": host_ram,
        "meminfo_after": after,
        "decode": {
            "prompt": args.prompt,
            "prompt_ids": prompt,
            "generated_ids": generated,
            "steps": steps,
        },
    }))
}

/// Compare part (c)'s decode with another run's JSON.
fn compare(mine: &Value, other_path: &Path) -> Result<Value, String> {
    let text = std::fs::read_to_string(other_path)
        .map_err(|e| format!("read {}: {e}", other_path.display()))?;
    let other: Value = serde_json::from_str(&text).map_err(|e| format!("parse compare: {e}"))?;
    let pick = |v: &Value| v["parts"]["c_full_load"]["decode"].clone();
    let (a, b) = (pick(mine), pick(&other));
    if a.is_null() || b.is_null() {
        return Err("both runs need a parts.c_full_load.decode section".into());
    }
    let ids_equal = a["prompt_ids"] == b["prompt_ids"] && a["generated_ids"] == b["generated_ids"];
    let hashes = |v: &Value| -> Vec<Value> {
        v["steps"]
            .as_array()
            .map(|s| s.iter().map(|x| x["logits_fnv64"].clone()).collect())
            .unwrap_or_default()
    };
    let (ha, hb) = (hashes(&a), hashes(&b));
    let first_divergent_step = ha.iter().zip(&hb).position(|(x, y)| x != y);
    let logits_equal = ha == hb && !ha.is_empty();
    Ok(json!({
        "against": other_path.display().to_string(),
        "other_source": other["parts"]["c_full_load"]["source"],
        "ids_equal": ids_equal,
        "logits_hashes_equal": logits_equal,
        "steps_compared": ha.len().min(hb.len()),
        "first_divergent_step": first_divergent_step,
        "pass": ids_equal && logits_equal,
    }))
}

fn run() -> Result<bool, String> {
    let args = parse_args()?;
    let mut hfq = HfqFile::open(&args.model).map_err(|e| format!("open model: {e:?}"))?;
    let mut cfg = DeepseekV4::config_from_hfq(&hfq)?;
    let mut gpu = Gpu::init().map_err(|e| format!("GPU init: {e:?}"))?;
    let mut report = json!({
        "model": args.model.display().to_string(),
        "layer": args.layer,
        "gpu_arch": gpu.arch.clone(),
        "model_content_digest": hex(&hfq.content_digest()),
        "parts": {},
    });
    let mut pass = true;
    let record = |report: &mut Value, name: &str, part: Result<Value, String>| -> bool {
        let value = match part {
            Ok(v) => v,
            Err(error) => json!({ "pass": false, "error": error }),
        };
        let ok = value["pass"].as_bool().unwrap_or(false);
        eprintln!("[{name}] {}", if ok { "PASS" } else { "FAIL" });
        report["parts"][name] = value;
        ok
    };

    if args.mapped {
        if args.layer >= cfg.num_hidden_layers {
            return Err(format!(
                "layer {} >= {} layers",
                args.layer, cfg.num_hidden_layers
            ));
        }
        let source = lab_mapped::trunk_source(&hfq, &cfg)?;
        // One group's admission: the same charge check the full load applies, for two blobs.
        let one_group = HostCharge {
            blob_bytes: cfg.n_routed_experts as u64 * source.expert_bytes().bundle(),
            tail_pad_bytes: 2 * HOST_TAIL_PAD_BYTES,
            allocations: 2,
        };
        check_host_ram(
            &one_group,
            hipfire_runtime::offload::pinnable_host_ram(Some(&gpu)),
        )
        .map_err(|r| r.to_string())?;
        let resident = lab_mapped::load_probe_layer(&hfq, &mut gpu, &cfg, args.layer, None)?;
        let mapped = lab_mapped::load_probe_layer(&hfq, &mut gpu, &cfg, args.layer, Some(&source))?;
        let a = part_a(&mut gpu, &hfq, &source, &args, &mapped);
        pass &= record(&mut report, "a_fetch_into_batch", a);
        write_report(&args.out, &report)?;
        let b = part_b(&mut gpu, &cfg, &resident, &mapped, &args);
        pass &= record(&mut report, "b_layer_ffn", b);
        resident.free(&mut gpu);
        mapped.free(&mut gpu);
        write_report(&args.out, &report)?;
    }

    if !args.skip_full_load {
        let c = part_c(&mut gpu, &mut hfq, &mut cfg, &args);
        pass &= record(&mut report, "c_full_load", c);
        write_report(&args.out, &report)?;
        if let Some(other) = &args.compare {
            let cmp = compare(&report, other);
            pass &= record(&mut report, "compare", cmp);
        }
    } else if args.compare.is_some() {
        return Err("--compare needs part (c); drop --skip-full-load".into());
    }
    report["pass"] = json!(pass);
    write_report(&args.out, &report)?;
    eprintln!("wrote {} (pass={pass})", args.out.display());
    Ok(pass)
}

fn main() {
    match run() {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(error) => {
            eprintln!("fn_ds4_mapped_parity: {error}");
            std::process::exit(2);
        }
    }
}
