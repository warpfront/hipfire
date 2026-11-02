//! CPU-only end-to-end check of a Flash-Next expert-slab artifact against the
//! qwen4 loader: parse `qwen4.expert_slabs`, build the span table from the real
//! index, validate every W3/W2 payload through the shared codec, then run the
//! production `admit_hfqm_artifact` and require its fail-closed
//! `ExpertSlabKernelsUnavailable` refusal. Never touches a GPU.
//!
//! The report counts experts per codec, per trunk layer and in total, for all
//! four codecs: MQ4 = codec 0, W3 = 1, W2 = 2, E8P = 3 (E8P-I4, qt135/136,
//! `ExpertFormat::E8pI4`). E8P payloads are span-checked only (the table
//! builder's exact length/offset checks); they are not decoded here.
//!
//! ```text
//! qwen4_slab_admit_check mixed.hfq > admit.json
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use clap::Parser;
use hipfire_arch_qwen4::artifact::{admit_hfqm_artifact, Qwen4ArtifactError};
use hipfire_arch_qwen4::expert_slab_table::{
    build_expert_slab_table_with_keep, validate_expert_slab_scales, ExpertSlabCounts,
    ExpertSlabTable,
};
use hipfire_arch_qwen4::expert_slabs::{self, ExpertFormat, EXPERTS_PER_LAYER};
use hipfire_arch_qwen4::ExpertKeep;
use hipfire_runtime::hfq::HfqFile;

#[derive(Parser, Debug)]
#[command(
    name = "qwen4_slab_admit_check",
    about = "Validate a Flash-Next expert-slab HFQ with the qwen4 loader (CPU only)"
)]
struct Args {
    /// Expert-slab HFQM artifact.
    path: PathBuf,
}

fn fail(stage: &str, error: impl std::fmt::Display) -> ExitCode {
    eprintln!("qwen4_slab_admit_check: {stage}: {error}");
    ExitCode::FAILURE
}

fn counts_json(counts: ExpertSlabCounts) -> serde_json::Value {
    serde_json::json!({"mq4": counts.mq4, "w3": counts.w3, "w2": counts.w2, "e8p": counts.e8p})
}

/// Experts per codec for every trunk layer plus the total, from the table's actual slab
/// formats (`layer * 512 + expert` order; the builder guarantees exactly 48 x 512) and
/// [`ExpertSlabTable::counts`]. `codec_ids` are the per-layer codec table bytes
/// (`rdna_compute::slab_admission`): MQ4 = 0, W3 (qt131/132) = 1, W2 (qt133/134) = 2 (no
/// route on any architecture), E8P (E8P-I4, qt135/136) = 3. Pure; no I/O.
fn codec_counts_json(table: &ExpertSlabTable) -> serde_json::Value {
    use rdna_compute::slab_admission::{SLAB_CODEC_E8P, SLAB_CODEC_MQ4, SLAB_CODEC_W3};
    let layers: Vec<_> = table
        .slabs()
        .chunks(EXPERTS_PER_LAYER as usize)
        .enumerate()
        .map(|(layer, slabs)| {
            let mut counts = ExpertSlabCounts::default();
            for slab in slabs {
                // Absent (pruned) original ids hold zero payload bytes and are in no codec's count.
                if slab.gate_up.len == 0 && slab.down.len == 0 {
                    continue;
                }
                match slab.format {
                    ExpertFormat::Mq4Xts => counts.mq4 += 1,
                    ExpertFormat::Mq3I4 => counts.w3 += 1,
                    ExpertFormat::Mq2I4 => counts.w2 += 1,
                    ExpertFormat::E8pI4 => counts.e8p += 1,
                }
            }
            let mut value = counts_json(counts);
            value["layer"] = layer.into();
            value
        })
        .collect();
    serde_json::json!({
        "codec_ids": {"mq4": SLAB_CODEC_MQ4, "w3": SLAB_CODEC_W3, "w2": 2, "e8p": SLAB_CODEC_E8P},
        "layers": layers,
        "total": counts_json(table.counts()),
    })
}

fn main() -> ExitCode {
    let args = Args::parse();
    let start = Instant::now();
    let hfq = match HfqFile::open(&args.path) {
        Ok(hfq) => hfq,
        Err(error) => return fail("open", error),
    };
    let metadata = match expert_slabs::parse(&hfq.metadata_json) {
        Ok(Some(metadata)) => metadata,
        Ok(None) => return fail("metadata", "no qwen4.expert_slabs key"),
        Err(error) => return fail("metadata", error),
    };
    let json: serde_json::Value = match serde_json::from_str(&hfq.metadata_json) {
        Ok(v) => v,
        Err(error) => return fail("keep metadata", error),
    };
    let keep = match ExpertKeep::from_metadata(&json) {
        Ok(keep) => keep,
        Err(error) => return fail("keep metadata", error),
    };
    for info in hfq.tensors() {
        if let Some((layer, expert, _)) = expert_slabs::parse_expert_entry_name(&info.name) {
            if keep.as_ref().is_some_and(|k| !k.is_kept(layer, expert)) {
                return fail("absent expert", format!("unexpected tensor {}", info.name));
            }
        }
    }
    let routed_bytes: usize = hfq
        .tensors()
        .iter()
        .filter(|t| expert_slabs::parse_expert_entry_name(&t.name).is_some())
        .map(|t| t.data_size)
        .sum();
    let non_expert_bytes: usize = hfq
        .tensors()
        .iter()
        .filter(|t| expert_slabs::parse_expert_entry_name(&t.name).is_none())
        .map(|t| t.data_size)
        .sum();
    let table = match build_expert_slab_table_with_keep(&metadata, hfq.tensors(), keep.as_ref()) {
        Ok(table) => table,
        Err(error) => return fail("span table", error),
    };
    let table_s = start.elapsed().as_secs_f64();
    let slab_entries: HashMap<usize, &str> = hfq
        .tensors()
        .iter()
        .filter(|info| expert_slabs::parse_expert_entry_name(&info.name).is_some())
        .map(|info| (info.data_offset, info.name.as_str()))
        .collect();
    if let Err(error) = validate_expert_slab_scales(&table, |span| {
        let name = slab_entries.get(&span.offset)?;
        let (info, bytes) = hfq.tensor_data(name)?;
        (info.data_size == span.len).then_some(bytes)
    }) {
        return fail("payload validation", error);
    }
    let validate_s = start.elapsed().as_secs_f64() - table_s;
    let counts = table.counts();
    if keep
        .as_ref()
        .is_some_and(|k| counts.w3 + counts.w2 + counts.e8p + counts.mq4 != k.total_kept())
    {
        return fail("keep counts", "table count differs from keep total");
    }
    match admit_hfqm_artifact(&hfq) {
        Err(Qwen4ArtifactError::ExpertSlabKernelsUnavailable { w3, w2, e8p, mq4 })
            if (w3, w2, e8p, mq4) == (counts.w3, counts.w2, counts.e8p, counts.mq4) =>
        {
            let refusal = Qwen4ArtifactError::ExpertSlabKernelsUnavailable { w3, w2, e8p, mq4 };
            let report = serde_json::json!({
                "artifact": args.path.display().to_string(),
                "tensors": hfq.tensors().len(),
                "expert_slabs": metadata.to_json_value(),
                "counts": {"w3": w3, "w2": w2, "e8p": e8p, "mq4": mq4},
                "codec_counts": codec_counts_json(&table),
                "expert_keep": keep.as_ref().map(ExpertKeep::to_metadata_value),
                "kept_experts": keep.as_ref().map_or(48 * 512, ExpertKeep::total_kept),
                "bytes": {"routed_experts": routed_bytes, "non_expert": non_expert_bytes, "total_base": hfq.file_len()},
                "payload_validation": "every W3/W2 projection decoded: exact length, codes <= L, finite non-negative f16 scales, no -0, zero-scale groups at the midpoint code; E8P-I4 (codec 3, qt=135/136) and MQ4 records are span-checked only here (exact index/offset/length by the span-table builder); E8P and MQ4 payload bytes are NOT decoded or validated by this tool",
                "admission": "refused (fail-closed)",
                "refusal": refusal.to_string(),
                "timings_s": {"open_and_table": table_s, "payload_validation": validate_s},
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&report).expect("report serializes")
            );
            ExitCode::SUCCESS
        }
        Err(error) => fail("admission returned a different refusal", error),
        Ok(_) => fail(
            "admission",
            "artifact was ADMITTED; expected the fail-closed kernel refusal",
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_arch_qwen4::artifact::admit_inventory_headers;
    use hipfire_arch_qwen4::config::ARCH_ID;
    use hipfire_arch_qwen4::expert_slab_table::build_expert_slab_table;
    use hipfire_arch_qwen4::expert_slabs::{
        expert_entry_name, ExpertFormat, ExpertProj, ExpertSlabsV1, EXPERT_SLABS_METADATA_KEY,
    };
    use hipfire_runtime::hfq::HfqTensorInfo;

    /// Synthetic mixed layout: layer 0 all MQ4; layer 1 MQ4 + E8P (every 4th
    /// expert E8P); layer 2 all E8P; layer 3 one quarter each of W3, W2, E8P
    /// and MQ4; every later layer all MQ4.
    fn format_at(layer: usize, expert: usize) -> ExpertFormat {
        match layer {
            1 if expert % 4 == 0 => ExpertFormat::E8pI4,
            2 => ExpertFormat::E8pI4,
            3 => match expert % 4 {
                0 => ExpertFormat::Mq3I4,
                1 => ExpertFormat::Mq2I4,
                2 => ExpertFormat::E8pI4,
                _ => ExpertFormat::Mq4Xts,
            },
            _ => ExpertFormat::Mq4Xts,
        }
    }

    /// Full 48x512 version-1 index (public APIs only): back-to-back spans from
    /// offset 4096, each entry at its format's exact payload size.
    fn synthetic_headers() -> Vec<HfqTensorInfo> {
        let mut out = Vec::with_capacity(48 * 512 * 2);
        let mut cursor = 4096usize;
        for layer in 0..48 {
            for expert in 0..512 {
                let format = format_at(layer, expert);
                let (gate_qt, down_qt) = format.pair();
                for (proj, qt, bytes) in [
                    (ExpertProj::GateUp, gate_qt, format.gate_up_bytes()),
                    (ExpertProj::Down, down_qt, format.down_bytes()),
                ] {
                    out.push(HfqTensorInfo {
                        name: expert_entry_name(layer, expert, proj),
                        quant_type: qt,
                        shape: proj.shape().to_vec(),
                        group_size: if qt == 44 { 256 } else { 128 },
                        data_offset: cursor,
                        data_size: bytes,
                    });
                    cursor += bytes;
                }
            }
        }
        out
    }

    fn layer_json(layer: usize, mq4: usize, w3: usize, w2: usize, e8p: usize) -> serde_json::Value {
        serde_json::json!({"layer": layer, "mq4": mq4, "w3": w3, "w2": w2, "e8p": e8p})
    }

    #[test]
    fn per_layer_codec_counts_on_synthetic_mixed_mq4_e8p_table() {
        let tensors = synthetic_headers();
        let table = build_expert_slab_table(&ExpertSlabsV1::fn_v1(), &tensors)
            .expect("synthetic mixed index builds through the production span-table builder");
        let json = codec_counts_json(&table);

        // Codec ids explicit: E8P = codec 3.
        assert_eq!(
            json["codec_ids"],
            serde_json::json!({"mq4": 0, "w3": 1, "w2": 2, "e8p": 3})
        );

        // Per layer.
        let layers = json["layers"].as_array().expect("layers array");
        assert_eq!(layers.len(), 48);
        assert_eq!(layers[0], layer_json(0, 512, 0, 0, 0));
        assert_eq!(layers[1], layer_json(1, 384, 0, 0, 128));
        assert_eq!(layers[2], layer_json(2, 0, 0, 0, 512));
        assert_eq!(layers[3], layer_json(3, 128, 128, 128, 128));
        for (layer, value) in layers.iter().enumerate().skip(4) {
            assert_eq!(*value, layer_json(layer, 512, 0, 0, 0));
        }

        // Total is the table's own counts(), and layers sum to it (every layer is 512).
        let table_counts = table.counts();
        assert_eq!(
            (
                table_counts.mq4,
                table_counts.w3,
                table_counts.w2,
                table_counts.e8p
            ),
            (23_552, 128, 128, 768)
        );
        assert_eq!(
            json["total"],
            serde_json::json!({"mq4": 23_552, "w3": 128, "w2": 128, "e8p": 768})
        );
        for codec in ["mq4", "w3", "w2", "e8p"] {
            let summed: u64 = layers.iter().map(|l| l[codec].as_u64().unwrap()).sum();
            assert_eq!(json["total"][codec].as_u64().unwrap(), summed, "{codec}");
        }
        for layer in layers {
            let sum: u64 = ["mq4", "w3", "w2", "e8p"]
                .iter()
                .map(|c| layer[c].as_u64().unwrap())
                .sum();
            assert_eq!(sum, 512);
        }

        // The production admission path recognizes the same index and its fail-closed
        // refusal tuple equals the table's counts (the tool's consistency condition).
        let metadata_json = serde_json::json!({
            "arch": "qwen4",
            EXPERT_SLABS_METADATA_KEY: ExpertSlabsV1::fn_v1().to_json_value(),
        })
        .to_string();
        match admit_inventory_headers(ARCH_ID, &metadata_json, &tensors) {
            Err(Qwen4ArtifactError::ExpertSlabKernelsUnavailable { w3, w2, e8p, mq4 }) => {
                assert_eq!((w3, w2, e8p, mq4), (128, 128, 768, 23_552));
                assert_eq!(
                    (w3, w2, e8p, mq4),
                    (
                        table_counts.w3,
                        table_counts.w2,
                        table_counts.e8p,
                        table_counts.mq4
                    )
                );
            }
            other => panic!(
                "expected the fail-closed ExpertSlabKernelsUnavailable refusal, got {other:?}"
            ),
        }
    }
}
