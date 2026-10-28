// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 Nick Woolmer
// Copyright (c) 2026 HUSRCF
// hipfire — see LICENSE and NOTICE in the project root.

//! Bounded vision-only companion pack. The official BF16 checkpoint stores its
//! tower AND merger in `model.visual.*`; the text producer excludes that group.
//! Only selected shard headers and visual payload ranges are fetched remotely.
//!
//! ```text
//! hipfire-quantize --qwen4-flash-next --vision-only \
//!   --input hf://Qwen/Qwen3.8-Flash-Next@de4b8e4d43b917e7706784d8bb445c9af86a3540 \
//!   --output qwen3.8-flash-next-vision.hfq
//! ```
//! A local BF16 checkpoint or visual safetensors subset is also accepted, with
//! the original `config.json` and `preprocessor_config.json` beside it.

use super::*;
use hipfire_quantize::vision_sidecar::{vision_dtype, VisionDtype};
use hipfire_runtime::hfq::{HfqFile, HfqStreamEntry, HfqTensorInfo};

const SOURCE_CHUNK_BYTES: usize = 8 * 1024 * 1024;

/// `--qwen4-flash-next --vision-only`: preserve the complete tower and merger
/// with the same F16-matrix/F32-vector policy as the existing Qwen35 sidecar.
/// The native text artifact and its pinned manifest are never rewritten.
pub(crate) fn run_vision_cli(input: &Path, output: &Path) -> Result<Qwen4Summary, Qwen4Error> {
    if output.exists() {
        return Err(Qwen4Error::Invalid(format!(
            "refusing to replace existing Qwen4 vision artifact {}",
            output.display()
        )));
    }
    let source = vision_source_paths(input)?;
    let config = load_optional_config(&source)?.ok_or_else(|| {
        Qwen4Error::Invalid("Qwen4 vision input is missing config.json".to_string())
    })?;
    let processor = match &source {
        SourceSet::Remote { source, .. } => {
            read_remote_optional_json(source, "preprocessor_config.json", MAX_CONFIG_BYTES)?
        }
        SourceSet::Local { paths, .. } => read_local_json(
            &paths[0]
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join("preprocessor_config.json"),
            MAX_CONFIG_BYTES,
        )?,
    };
    let tensors: Vec<_> = load_inventory(&source)?
        .into_iter()
        .filter(|tensor| is_vision_tensor(&tensor.name))
        .collect();
    let entries = plan_vision_entries(&tensors)?;
    // Fail on a foreign or incomplete tower before reading any weight payload.
    let geometry =
        hipfire_arch_qwen4::vision::config_from_json(&config).map_err(Qwen4Error::Invalid)?;
    let tensor_infos: Vec<_> = entries
        .iter()
        .map(|entry| HfqTensorInfo {
            name: entry.name.clone(),
            quant_type: entry.quant_type,
            shape: entry.shape.clone(),
            group_size: entry.group_size,
            data_offset: 0,
            data_size: entry.data_len as usize,
        })
        .collect();
    hipfire_arch_qwen4::vision::validate_tensor_infos(&tensor_infos, &geometry)
        .map_err(Qwen4Error::Invalid)?;
    let provenance = match &source {
        SourceSet::Remote { source, paths } => json!({
            "repository": format!("{}/{}", source.spec.owner, source.spec.repo),
            "revision": source.spec.revision,
            "shards": paths,
        }),
        SourceSet::Local { .. } => json!({"input": input.to_string_lossy()}),
    };
    let metadata_json = serde_json::to_string(&json!({
        "config": config,
        "preprocessor_config": processor,
        "has_vision": true,
        "hipfire_vision_only": true,
        "hipfire_source": provenance,
    }))
    .map_err(|error| Qwen4Error::json("serialize Qwen4 vision metadata", error))?;
    let predicted_output_bytes = predicted_output_bytes(&metadata_json, &entries)?;
    capacity_preflight(output, predicted_output_bytes)?;
    let temporary = temporary_output_path(output)?;
    let mut scratch_high_water_bytes = 0;
    let result = (|| {
        hipfire_runtime::hfq::write_hfqm_package_streaming(
            &temporary,
            QWEN4_ARCH_ID,
            &metadata_json,
            &entries,
            |index, writer| {
                eprintln!(
                    "qwen4 vision [{}/{}]: {} ({} source bytes)",
                    index + 1,
                    tensors.len(),
                    tensors[index].name,
                    tensors[index].data_len()
                );
                let allocated =
                    stream_vision_tensor(&tensors[index], entries[index].quant_type, writer)
                        .map_err(Qwen4Error::into_io)?;
                scratch_high_water_bytes = scratch_high_water_bytes.max(allocated);
                Ok(())
            },
        )
        .map_err(|error| Qwen4Error::io("write Qwen4 vision artifact", error))?;
        sync_file(&temporary)?;
        source.verify_identity()?;
        let reopened = HfqFile::open(&temporary)
            .map_err(|error| Qwen4Error::io("reopen Qwen4 vision artifact", error))?;
        let config =
            hipfire_arch_qwen4::vision::config_from_hfq(&reopened).map_err(Qwen4Error::Invalid)?;
        hipfire_arch_qwen4::vision::validate_hfq(&reopened, &config)
            .map_err(Qwen4Error::Invalid)?;
        drop(reopened);
        publish_qwen4(&temporary, output)
    })();
    if let Err(error) = result {
        let _ = fs::remove_file(&temporary);
        return Err(error);
    }
    Ok(Qwen4Summary {
        entries: entries.len(),
        resident_entries: entries.len(),
        expert_entries: 0,
        ple_shards: 0,
        resident_bytes: entries.iter().map(|entry| entry.data_len).sum(),
        external_ple_bytes: 0,
        predicted_output_bytes,
        scratch_high_water_bytes,
    })
}

fn vision_source_paths(input: &Path) -> Result<SourceSet, Qwen4Error> {
    if let Some(spec) = input.to_str().map(parse_remote_spec).transpose()?.flatten() {
        let source = Arc::new(RemoteSource::new(spec)?);
        let index = source.read_json("model.safetensors.index.json", MAX_CONFIG_BYTES)?;
        let paths = vision_shards_from_index(&index)?;
        return Ok(SourceSet::Remote { source, paths });
    }
    source_paths(input)
}

fn vision_shards_from_index(index: &Value) -> Result<Vec<String>, Qwen4Error> {
    let weight_map = index
        .get("weight_map")
        .and_then(Value::as_object)
        .ok_or_else(|| Qwen4Error::Invalid("safetensors index has no weight_map".to_string()))?;
    let mut paths = BTreeSet::new();
    let mut selected = 0;
    for (name, path) in weight_map {
        if !is_vision_tensor(name) {
            continue;
        }
        let path = path.as_str().ok_or_else(|| {
            Qwen4Error::Invalid(format!("vision tensor {name} has a non-string shard"))
        })?;
        validate_remote_path(path, Some("safetensors"))?;
        paths.insert(path.to_string());
        selected += 1;
    }
    if selected != PINNED_VISION_TENSOR_COUNT {
        return Err(Qwen4Error::Invalid(format!(
            "Qwen4 vision index requires {PINNED_VISION_TENSOR_COUNT} tensors, found {selected}"
        )));
    }
    Ok(paths.into_iter().collect())
}

fn plan_vision_entries(tensors: &[SourceTensor]) -> Result<Vec<HfqStreamEntry>, Qwen4Error> {
    if tensors.len() != PINNED_VISION_TENSOR_COUNT {
        return Err(Qwen4Error::Invalid(format!(
            "Qwen4 vision inventory requires {PINNED_VISION_TENSOR_COUNT} tensors, found {}",
            tensors.len()
        )));
    }
    tensors
        .iter()
        .map(|tensor| {
            if tensor.dtype != "BF16" {
                return Err(Qwen4Error::Invalid(format!(
                    "Qwen4 vision source {} must be BF16, found {}",
                    tensor.name, tensor.dtype
                )));
            }
            let (quant_type, element_bytes) = match vision_dtype(&tensor.name, tensor.shape.len()) {
                Some(VisionDtype::F16Matrix) => (1, 2),
                Some(VisionDtype::F32Vector) => (2, 4),
                None => {
                    return Err(Qwen4Error::Invalid(format!(
                        "no vision dtype for {}",
                        tensor.name
                    )));
                }
            };
            let shape = tensor
                .shape
                .iter()
                .map(|&dimension| {
                    u32::try_from(dimension).map_err(|_| {
                        Qwen4Error::Invalid(format!("{} dimension exceeds u32", tensor.name))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let data_len = checked_product(&tensor.shape, &tensor.name)?
                .checked_mul(element_bytes)
                .ok_or_else(|| Qwen4Error::Invalid("vision payload size overflow".to_string()))?;
            Ok(HfqStreamEntry {
                name: tensor.name.clone(),
                quant_type,
                shape,
                group_size: 0,
                data_len,
            })
        })
        .collect()
}

fn stream_vision_tensor(
    tensor: &SourceTensor,
    quant_type: u8,
    writer: &mut dyn Write,
) -> Result<u64, Qwen4Error> {
    let mut offset = 0;
    let mut source = vec![0u8; SOURCE_CHUNK_BYTES.min(tensor.data_len() as usize)];
    let mut encoded = Vec::with_capacity(source.len() * if quant_type == 2 { 2 } else { 1 });
    while offset < tensor.data_len() {
        let count = (tensor.data_len() - offset).min(source.len() as u64) as usize;
        tensor.read_range(offset, &mut source[..count])?;
        encoded.clear();
        for bytes in source[..count].chunks_exact(2) {
            let value = bf16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]));
            if quant_type == 2 {
                encoded.extend_from_slice(&value.to_le_bytes());
            } else {
                encoded.extend_from_slice(&half::f16::from_f32(value).to_bits().to_le_bytes());
            }
        }
        writer.write_all(&encoded).map_err(|error| {
            Qwen4Error::io(format!("write vision tensor {}", tensor.name), error)
        })?;
        offset += count as u64;
    }
    Ok((source.capacity() + encoded.capacity()) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remote_selection_reads_only_visual_shards_and_retains_merger() {
        let mut weight_map = Map::new();
        for i in 0..PINNED_VISION_TENSOR_COUNT - 1 {
            weight_map.insert(
                format!("model.visual.block.{i}"),
                json!("vision.safetensors"),
            );
        }
        weight_map.insert(
            "model.visual.merger.weight".into(),
            json!("merger.safetensors"),
        );
        weight_map.insert(
            "model.language_model.weight".into(),
            json!("huge-text.safetensors"),
        );
        let index = json!({"weight_map": weight_map});
        assert_eq!(
            vision_shards_from_index(&index).unwrap(),
            ["merger.safetensors", "vision.safetensors"]
        );
        let mut incomplete = index;
        incomplete["weight_map"]
            .as_object_mut()
            .unwrap()
            .remove("model.visual.merger.weight");
        assert!(vision_shards_from_index(&incomplete).is_err());
    }

    #[test]
    fn conversion_preserves_bf16_vectors_and_uses_f16_matrix_container() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("weights.safetensors");
        let values = [0x3f80u16, 0xc020, 0x3eab, 0x8000];
        let payload: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let header = serde_json::to_vec(&json!({"model.visual.norm.weight": {"dtype": "BF16", "shape": [4], "data_offsets": [0, 8]}})).unwrap();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(payload);
        fs::write(&path, bytes).unwrap();
        let tensor = parse_safetensors_header(&path).unwrap().remove(0);
        for (quant_type, expected) in [
            (1, vec![0x00, 0x3c, 0x00, 0xc1, 0x58, 0x35, 0x00, 0x80]),
            (
                2,
                vec![
                    0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x20, 0xc0, 0x00, 0x00, 0xab, 0x3e, 0x00,
                    0x00, 0x00, 0x80,
                ],
            ),
        ] {
            let mut emitted = Vec::new();
            stream_vision_tensor(&tensor, quant_type, &mut emitted).unwrap();
            assert_eq!(emitted, expected);
        }
    }

    #[test]
    fn existing_artifact_is_preserved_before_opening_source() {
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("vision.hfq");
        fs::write(&output, b"existing artifact").unwrap();
        assert!(run_vision_cli(&dir.path().join("missing"), &output).is_err());
        assert_eq!(fs::read(output).unwrap(), b"existing artifact");
    }
}
