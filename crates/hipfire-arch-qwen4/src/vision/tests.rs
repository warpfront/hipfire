// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 HUSRCF
// hipfire — see LICENSE and NOTICE in the project root.

use super::*;

fn source_config() -> Value {
    serde_json::from_str(include_str!(
        "../../tests/fixtures/qwen4_vision_config.json"
    ))
    .unwrap()
}

// Independent fixture: shapes copied from the official safetensors header at
// the revision named in the JSON, not generated from expected_shapes().
fn source_inventory() -> Vec<HfqTensorInfo> {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/qwen4_vision_shapes.json"
    ))
    .unwrap();
    fixture["shapes"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, shape)| {
            let shape: Vec<u32> = serde_json::from_value(shape.clone()).unwrap();
            let quant_type = if shape.len() == 1 { 2 } else { 1 };
            let bytes = if quant_type == 1 { 2 } else { 4 };
            HfqTensorInfo {
                name: name.clone(),
                quant_type,
                data_size: shape.iter().fold(bytes, |n, d| n * *d as usize),
                shape,
                group_size: 0,
                data_offset: 0,
            }
        })
        .collect()
}

#[test]
fn official_flash_next_tower_matches_shared_tower_contract() {
    let config = config_from_json(&source_config()).unwrap();
    let inventory = source_inventory();
    assert_eq!(inventory.len(), 333);
    validate_tensor_infos(&inventory, &config).unwrap();
    assert_eq!(config.out_hidden_size, 2560);
    let resident: usize = inventory
        .iter()
        .filter(|t| t.name != "model.visual.pos_embed.weight")
        .map(|t| t.data_size)
        .sum();
    assert_eq!(resident as u64, DEVICE_WEIGHT_BYTES);
    // Small images follow the official processor minimum, not the old
    // Qwen2 patch-14 policy: a 32x32 image becomes a 16x16 patch grid.
    assert_eq!(
        crate::image::smart_resize(32, 32, 32, 65_536, 2_000_000),
        (256, 256)
    );
}

#[test]
fn refuses_wrong_model_projector_or_image_contract() {
    for (section, key, replacement) in [
        ("vision_config", "out_hidden_size", serde_json::json!(5120)),
        ("vision_config", "num_heads", serde_json::json!(0)),
        (
            "vision_config",
            "deepstack_visual_indexes",
            serde_json::json!([8, 16, 24]),
        ),
        ("vision_config", "hidden_act", serde_json::json!("gelu")),
        ("text_config", "hidden_size", serde_json::json!(5120)),
    ] {
        let mut config = source_config();
        config[section][key] = replacement;
        assert!(
            config_from_json(&config).is_err(),
            "accepted {section}.{key}"
        );
    }
    let mut config = source_config();
    config["image_token_id"] = serde_json::json!(151655);
    assert!(config_from_json(&config)
        .unwrap_err()
        .contains("image_token_id"));
}

#[test]
fn refuses_missing_or_foreign_or_wrong_shape_tensor_before_upload() {
    let config = config_from_json(&source_config()).unwrap();
    let mut inventory = source_inventory();
    inventory.pop();
    assert!(validate_tensor_infos(&inventory, &config)
        .unwrap_err()
        .contains("333"));
    let mut inventory = source_inventory();
    let projector = inventory
        .iter_mut()
        .find(|t| t.name == "model.visual.merger.linear_fc2.weight")
        .unwrap();
    projector.shape[0] = 5120;
    assert!(validate_tensor_infos(&inventory, &config)
        .unwrap_err()
        .contains("shape"));
    let mut inventory = source_inventory();
    inventory[0].name = "model.vision_projection.weight".into();
    assert!(validate_tensor_infos(&inventory, &config)
        .unwrap_err()
        .contains("unexpected"));
    let mut inventory = source_inventory();
    inventory[1] = inventory[0].clone();
    assert!(validate_tensor_infos(&inventory, &config)
        .unwrap_err()
        .contains("duplicate"));
}

#[test]
fn refuses_truncated_or_unadmitted_quant_payload_before_upload() {
    let config = config_from_json(&source_config()).unwrap();
    let mut inventory = source_inventory();
    inventory[0].data_size -= 1;
    assert!(validate_tensor_infos(&inventory, &config)
        .unwrap_err()
        .contains("bytes"));
    let mut inventory = source_inventory();
    inventory[0].quant_type = 13;
    assert!(validate_tensor_infos(&inventory, &config)
        .unwrap_err()
        .contains("unsupported quant"));
}

#[test]
fn scratch_admission_includes_pool_retention_and_refuses_unbounded_images() {
    assert_eq!(device_scratch_bytes(16_128, true).unwrap(), 1_219_534_848);
    assert_eq!(device_scratch_bytes(16_128, false).unwrap(), 1_484_808_192);
    assert!(device_scratch_bytes(0, true).is_err());
    assert!(device_scratch_bytes(16_132, true).is_err());
    assert!(device_scratch_bytes(5, true).is_err());
}
