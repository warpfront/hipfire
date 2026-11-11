//! Independent, fail-closed evidence checks for the MTP/LRU developer oracle.
//! Synthetic controls below qualify these checks only, never a GPU or model.

use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

pub(super) type R<T> = Result<T, String>;
pub(super) const LAYERS: usize = 48;
pub(super) const TOPK: usize = 10;
pub(super) const EXPERTS: usize = 512;
pub(super) const MAX_ROWS: usize = 8;

pub(super) fn ensure(ok: bool, message: impl Into<String>) -> R<()> {
    if ok {
        Ok(())
    } else {
        Err(message.into())
    }
}
pub(super) fn number(v: &Value, key: &str) -> R<u64> {
    v.get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("missing unsigned {key}"))
}
pub(super) fn route_pattern(layer: usize, rows: usize) -> Vec<i32> {
    (0..rows)
        .flat_map(|row| (0..TOPK).map(move |k| ((71 * layer + 43 * row + 17 * k) % EXPERTS) as i32))
        .collect()
}
pub(super) fn expected_routes(rows: usize) -> R<Vec<i32>> {
    ensure(
        (1..=MAX_ROWS).contains(&rows),
        "collector row extent outside 1..=8",
    )?;
    Ok((0..LAYERS)
        .flat_map(|layer| route_pattern(layer, rows))
        .collect())
}
pub(super) fn check_routes(rows: usize, actual: &[i32]) -> R<()> {
    let expected = expected_routes(rows)?;
    ensure(
        actual.len() == expected.len(),
        "collector did not return every layer/row/top-k",
    )?;
    if let Some(i) = actual.iter().zip(&expected).position(|(a, b)| a != b) {
        return Err(format!(
            "collector mismatch layer={} row={} k={}: actual={} expected={}",
            i / (rows * TOPK),
            (i / TOPK) % rows,
            i % TOPK,
            actual[i],
            expected[i]
        ));
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub(super) struct Slot {
    pub layer: usize,
    pub slot: u64,
    pub global_slot: u64,
    pub expert: u64,
    pub generation: u64,
    pub epoch: u64,
    pub gu: u64,
    pub dn: u64,
}
/// Actual SlotBank identity, independent of additional diagnostic JSON fields.
/// Local slot numbers repeat between layers; global_slot must never alias them.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct Ticket {
    pub layer: u64,
    pub slot: u64,
    pub global_slot: u64,
    pub expert: u64,
    pub generation: u64,
}
pub(super) fn ticket(v: &Value) -> R<Ticket> {
    let t = Ticket {
        layer: number(v, "layer")?,
        slot: number(v, "slot")?,
        global_slot: number(v, "global_slot")?,
        expert: number(v, "expert")?,
        generation: number(v, "generation")?,
    };
    ensure(
        t.layer < LAYERS as u64 && t.expert < EXPERTS as u64 && t.generation > 0,
        "out-of-range SlotBank ticket",
    )?;
    Ok(t)
}
pub(super) fn pending(v: &Value) -> R<BTreeSet<Ticket>> {
    let entries = v["pending"]
        .as_array()
        .ok_or("missing actual PendingFills tickets")?;
    let mut result = BTreeSet::new();
    let mut globals = BTreeSet::new();
    for p in entries {
        let t = ticket(p)?;
        ensure(
            globals.insert(t.global_slot),
            "one global SlotBank cell has two pending fills",
        )?;
        for field in ["gate_up", "down", "gate_up_bytes", "down_bytes"] {
            ensure(
                number(p, field)? > 0,
                format!("pending ticket lacks {field}"),
            )?;
        }
        ensure(result.insert(t), "duplicate actual pending ticket")?;
    }
    Ok(result)
}

/// The complete canonical receipt includes transaction capture buffers and a
/// physical ring index. Their bytes are retained separately, not compared
/// across different AR/verify schedules. Every other live row remains strict.
pub(super) fn target_projection(
    rows: Vec<(String, String)>,
    gdn_layers: &[usize],
    residual: bool,
) -> R<(Vec<(String, String)>, Vec<(String, String)>)> {
    let mut all = BTreeMap::new();
    for (key, value) in rows {
        ensure(
            all.insert(key, value).is_none(),
            "duplicate canonical target state row",
        )?;
    }
    ensure(!all.is_empty(), "missing canonical target state")?;
    for layer in gdn_layers {
        for field in ["recurrent", "conv"] {
            ensure(
                all.contains_key(&format!("gdn[L{layer}].{field}")),
                format!("missing actual GDN layer {layer} {field}"),
            )?;
        }
        if residual {
            ensure(
                all.contains_key(&format!("gdn[L{layer}].ef_residual")),
                format!("missing actual GDN layer {layer} EF residual"),
            )?;
        }
    }
    let physical: BTreeSet<_> = gdn_layers
        .iter()
        .flat_map(|layer| {
            [
                format!("gdn[L{layer}].live"),
                format!("row_capture[L{layer}].ring"),
                format!("row_capture[L{layer}].inputs"),
                format!("row_capture[L{layer}].recurrence"),
            ]
        })
        .collect();
    Ok(all
        .into_iter()
        .partition(|(key, _)| !physical.contains(key)))
}
fn array<'a>(v: &'a Value, key: &str, len: usize) -> R<&'a Vec<Value>> {
    let a = v
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing array {key}"))?;
    ensure(
        a.len() == len,
        format!("{key}: expected {len}, got {}", a.len()),
    )?;
    Ok(a)
}
/// Read actual published tables rather than treating an eviction reservation as a fill.
pub(super) fn published(v: &Value) -> R<Vec<Slot>> {
    ensure(
        v["schema"] == "hipfire-expert-cache-snapshot-v1",
        "cache snapshot schema",
    )?;
    ensure(
        v["slot_authority"] == "railgun::sealed_table::generation::SlotBank",
        "snapshot is not backed by actual SlotBank cells",
    )?;
    let _ = number(v, "bank_epoch")?;
    ensure(v["poisoned"].is_null(), "cache poisoned")?;
    ensure(
        v["outstanding"].as_bool() == Some(false),
        "unretired batch at observation boundary",
    )?;
    ensure(number(v, "rows_cap")? == MAX_ROWS as u64, "cache row cap")?;
    let layers = array(v, "layers", LAYERS)?;
    let mut seen_layers = BTreeSet::new();
    let mut seen_globals = BTreeSet::new();
    let mut result = Vec::new();
    for l in layers {
        let layer = number(l, "layer")? as usize;
        ensure(
            layer < LAYERS && seen_layers.insert(layer),
            "duplicate/out-of-range cache layer",
        )?;
        let epoch = number(l, "epoch")?;
        let gu = array(l, "gate_up", EXPERTS)?;
        let dn = array(l, "down", EXPERTS)?;
        let canonical_gu = array(l, "canonical_gate_up", EXPERTS)?;
        let canonical_dn = array(l, "canonical_down", EXPERTS)?;
        let mappings = array(l, "mapping", EXPERTS)?;
        let mut used_slots = BTreeSet::new();
        for (expert, m) in mappings.iter().enumerate() {
            let g = gu[expert].as_u64().ok_or("invalid GU address")?;
            let d = dn[expert].as_u64().ok_or("invalid DN address")?;
            ensure(g != 0 && d != 0, "null expert address")?;
            if m.is_null() {
                ensure(
                    gu[expert] == canonical_gu[expert] && dn[expert] == canonical_dn[expert],
                    "canonical fallback does not match BOTH host mirrors",
                )?;
                continue;
            }
            let slot = number(m, "slot")?;
            let global_slot = number(m, "global_slot")?;
            let generation = number(m, "generation")?;
            ensure(
                number(m, "expert")? == expert as u64 && generation > 0,
                "mapping expert/generation mismatch",
            )?;
            ensure(
                used_slots.insert(slot),
                "one physical slot published for two experts",
            )?;
            ensure(
                seen_globals.insert(global_slot),
                "global SlotBank cell published twice",
            )?;
            ensure(
                number(m, "gate_up")? == g && number(m, "down")? == d,
                "published tables differ from actual SlotBank pair",
            )?;
            ensure(
                number(m, "gate_up_bytes")? > 0 && number(m, "down_bytes")? > 0,
                "published SlotBank payload extent missing",
            )?;
            ensure(
                gu[expert] != canonical_gu[expert] && dn[expert] != canonical_dn[expert],
                "half-published slot pair",
            )?;
            result.push(Slot {
                layer,
                slot,
                global_slot,
                expert: expert as u64,
                generation,
                epoch,
                gu: g,
                dn: d,
            });
        }
    }
    for p in pending(v)? {
        ensure(
            !seen_globals.contains(&p.global_slot),
            "pending cell still published",
        )?;
    }
    Ok(result)
}
pub(super) fn turnover(before: &Value, after: &Value) -> R<Vec<Value>> {
    let old = published(before)?;
    let new = published(after)?;
    for key in ["selected", "fills_completed", "publish_calls"] {
        ensure(
            number(&after["stats"], key)? > number(&before["stats"], key)?,
            format!("native interval did not increase {key}"),
        )?;
    }
    let old: BTreeMap<_, _> = old.into_iter().map(|s| ((s.layer, s.slot), s)).collect();
    let mut changes = Vec::new();
    for n in new {
        if let Some(o) = old.get(&(n.layer, n.slot)) {
            ensure(
                n.generation >= o.generation,
                "slot generation moved backwards",
            )?;
            ensure(n.gu == o.gu && n.dn == o.dn, "physical slot pair moved")?;
            ensure(
                n.global_slot == o.global_slot,
                "layer/local slot changed SlotBank identity",
            )?;
            if n.expert != o.expert {
                ensure(
                    n.generation > o.generation && n.epoch > o.epoch,
                    "reused slot lacks new completed generation/epoch",
                )?;
                changes.push(json!({"layer":n.layer,"slot":n.slot,"global_slot":n.global_slot,"from_expert":o.expert,"to_expert":n.expert,"generation_before":o.generation,"generation_after":n.generation,"epoch_before":o.epoch,"epoch_after":n.epoch,"gate_up":n.gu,"down":n.dn}));
            } else {
                ensure(
                    n.generation == o.generation || n.epoch > o.epoch,
                    "reloaded expert lacks a newer publication epoch",
                )?;
            }
        }
    }
    ensure(
        !changes.is_empty(),
        "no completed, paired physical-slot victim turnover observed",
    )?;
    Ok(changes)
}
pub(super) fn real_native_window(v: &Value) -> R<()> {
    ensure(v["source"] == "Mtp", "window did not execute native MTP")?;
    ensure(
        matches!(v["route"].as_str(), Some("Batched" | "Interleaved")),
        "native window executed target-only fallback",
    )?;
    ensure(
        v["outcome_mismatch"].as_bool() == Some(false),
        "forced request did not execute requested native source/width",
    )?;
    let drafts = number(v, "actual_drafts")?;
    ensure(
        drafts > 0 && drafts == number(v, "requested_drafts")?,
        "no real native drafts or wrong width",
    )?;
    ensure(
        number(v, "accepted_prefix")? <= drafts,
        "impossible native acceptance",
    )?;
    ensure(
        number(v, "committed")? > 0,
        "native transaction committed no tokens",
    )?;
    let committed = number(v, "committed")?;
    ensure(
        committed == number(v, "accepted_prefix")? + 1,
        "native pending-seed committed count mismatch",
    )?;
    ensure(
        number(v, "position_after")?.checked_sub(number(v, "position_before")?) == Some(committed),
        "native target/committed advance mismatch",
    )
}

/// Deliberate synthetic controls qualify the checker, never a model or GPU.
pub(super) fn cpu_controls() -> R<Value> {
    let mut cases = Vec::new();
    for rows in 1..=MAX_ROWS {
        let good = expected_routes(rows)?;
        check_routes(rows, &good)?;
        let mut bad = good.clone();
        bad.truncate(bad.len() - 1);
        ensure(
            check_routes(rows, &bad).is_err(),
            "missing route escaped checker",
        )?;
        let mut bad = good.clone();
        let at = TOPK.min(bad.len() - 1);
        bad.swap(0, at);
        ensure(
            check_routes(rows, &bad).is_err(),
            "layer/row transposition escaped checker",
        )?;
        cases.push(
            json!({"case":"collector_missing_and_transposed_ids","rows":rows,"status":"PASS"}),
        );
    }
    let good = json!({"source":"Mtp","route":"Batched","requested_drafts":3,"actual_drafts":3,"accepted_prefix":1,"committed":2,"outcome_mismatch":false,"position_before":100,"position_after":102});
    real_native_window(&good)?;
    for (key, value) in [
        ("source", json!("ArRouted")),
        ("route", json!("TargetOnly")),
        ("actual_drafts", json!(0)),
        ("committed", json!(0)),
        ("outcome_mismatch", json!(true)),
        ("accepted_prefix", json!(4)),
    ] {
        let mut bad = good.clone();
        bad[key] = value;
        ensure(
            real_native_window(&bad).is_err(),
            format!("invalid native field {key} escaped checker"),
        )?;
        cases.push(json!({"case":format!("native_{key}"),"status":"PASS"}));
    }
    Ok(
        json!({"scope":"synthetic-receipt-negative-controls-only","gpu_opened":false,"model_loaded":false,"gpu_or_model_qualification":false,"cases":cases}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    fn snapshot(expert: usize, generation: u64, epoch: u64, count: u64) -> Value {
        let layers:Vec<_>=(0..LAYERS).map(|layer| {
            let cg:Vec<_>=(0..EXPERTS).map(|e| 0x100000u64+layer as u64*0x10000+e as u64*16).collect();
            let cd:Vec<_>=cg.iter().map(|x| x+0x10000000).collect();
            let mut gu=cg.clone(); let mut dn=cd.clone();
            let mut mapping=vec![Value::Null;EXPERTS];
            gu[expert]=0x50000000+layer as u64*0x1000;
            dn[expert]=0x60000000+layer as u64*0x1000;
            mapping[expert]=json!({"slot":0,"global_slot":layer,"expert":expert,"generation":generation,"gate_up":gu[expert],"down":dn[expert],"gate_up_bytes":16,"down_bytes":16});
            json!({"layer":layer,"epoch":epoch,"canonical_gate_up":cg,"canonical_down":cd,"gate_up":gu,"down":dn,"mapping":mapping})
        }).collect();
        json!({"schema":"hipfire-expert-cache-snapshot-v1","slot_authority":"railgun::sealed_table::generation::SlotBank","bank_epoch":epoch,"rows_cap":8,"outstanding":false,"poisoned":null,"pending":[],"stats":{"selected":count,"fills_completed":count,"publish_calls":count,"evictions":count},"layers":layers})
    }
    fn native() -> Value {
        json!({"source":"Mtp","route":"Batched","requested_drafts":3,"actual_drafts":3,"accepted_prefix":1,"committed":2,"outcome_mismatch":false,"position_before":100,"position_after":102})
    }
    #[test]
    fn collector_detects_row_alias_missing_layer_and_topk_transpose() {
        for rows in [1, 2, 3, 4, 7, 8] {
            let good = expected_routes(rows).unwrap();
            check_routes(rows, &good).unwrap();
            let mut bad = good.clone();
            bad.truncate(bad.len() - TOPK);
            assert!(check_routes(rows, &bad).is_err());
            let mut bad = good.clone();
            let at = TOPK.min(bad.len() - 1);
            bad.swap(0, at);
            assert!(check_routes(rows, &bad).is_err());
            if rows > 1 {
                let mut bad = good.clone();
                bad.copy_within(0..TOPK, TOPK);
                assert!(check_routes(rows, &bad).is_err());
            }
        }
    }
    #[test]
    fn native_receipt_refuses_ar_fallback_zero_drafts_and_uncommitted_window() {
        real_native_window(&native()).unwrap();
        for (key, value) in [
            ("source", json!("ArRouted")),
            ("route", json!("TargetOnly")),
            ("actual_drafts", json!(0)),
            ("committed", json!(0)),
            ("outcome_mismatch", json!(true)),
            ("accepted_prefix", json!(4)),
        ] {
            let mut bad = native();
            bad[key] = value;
            assert!(real_native_window(&bad).is_err(), "{key}");
        }
    }
    #[test]
    fn counter_only_eviction_cannot_prove_published_turnover() {
        let before = snapshot(0, 1, 1, 1);
        let mut after = snapshot(0, 1, 1, 2);
        after["stats"]["evictions"] = json!(1000);
        assert!(turnover(&before, &after).is_err());
        assert_eq!(
            turnover(&before, &snapshot(1, 2, 2, 2)).unwrap().len(),
            LAYERS
        );
    }
    #[test]
    fn half_publication_generation_rewind_and_live_guard_are_rejected() {
        let before = snapshot(0, 2, 3, 1);
        for case in 0..5 {
            let mut after = snapshot(1, 3, 4, 2);
            match case {
                0 => {
                    after["layers"][0]["down"][1] = after["layers"][0]["canonical_down"][1].clone()
                }
                1 => after["layers"][0]["mapping"][1]["generation"] = json!(1),
                2 => after["outstanding"] = json!(true),
                3 => after["poisoned"] = json!("unretired batch"),
                _ => after["layers"][0]["gate_up"][1] = json!(0x1234),
            }
            assert!(turnover(&before, &after).is_err(), "case {case}");
        }
    }

    #[test]
    fn global_bank_cell_cannot_alias_layers_pending_or_published_addresses() {
        published(&snapshot(0, 1, 1, 1)).unwrap();
        let mut bad = snapshot(0, 1, 1, 1);
        bad["layers"][1]["mapping"][0]["global_slot"] = json!(0);
        assert!(published(&bad).is_err());
        let mut bad = snapshot(0, 1, 1, 1);
        bad["layers"][0]["mapping"][0]["gate_up"] = json!(1);
        assert!(published(&bad).is_err());
        let mut bad = snapshot(0, 1, 1, 1);
        bad["pending"] = json!([{"layer":0,"slot":0,"global_slot":0,"expert":7,
            "generation":2,"gate_up":8,"down":16,"gate_up_bytes":8,"down_bytes":8}]);
        assert!(published(&bad).is_err());
        let mut bad = snapshot(0, 1, 1, 1);
        bad["slot_authority"] = json!("counter-only");
        assert!(published(&bad).is_err());
    }

    #[test]
    fn pending_identity_uses_real_bank_generation_not_json_diagnostic_equality() {
        let original = json!({"layer":3,"slot":0,"global_slot":91,"expert":7,"generation":9});
        let mut extended = original.clone();
        extended["gate_up"] = json!(0x1000);
        extended["diagnostic"] = json!("additional checked payload metadata");
        assert_eq!(ticket(&original).unwrap(), ticket(&extended).unwrap());
        for (key, value) in [
            ("layer", 4),
            ("slot", 1),
            ("global_slot", 92),
            ("expert", 8),
            ("generation", 10),
        ] {
            let mut changed = extended.clone();
            changed[key] = json!(value);
            assert_ne!(
                ticket(&original).unwrap(),
                ticket(&changed).unwrap(),
                "{key}"
            );
        }
    }

    #[test]
    fn target_projection_keeps_actual_recurrent_conv_residual_and_detects_missing_parts() {
        let state = vec![
            ("layer[L0].position".into(), "10".into()),
            ("gdn[L0].live".into(), "0".into()),
            ("gdn[L0].recurrent".into(), "recurrent bytes=64".into()),
            ("gdn[L0].conv".into(), "conv bytes=64".into()),
            ("gdn[L0].ef_residual".into(), "residual bytes=64".into()),
            ("row_capture[L0].ring".into(), "ring bytes=128".into()),
            ("row_capture[L0].inputs".into(), "inputs bytes=128".into()),
            (
                "row_capture[L0].recurrence".into(),
                "recurrence bytes=128".into(),
            ),
        ];
        let (logical, physical) = target_projection(state.clone(), &[0], true).unwrap();
        assert_eq!(physical.len(), 4);
        let mut other = state.clone();
        other[1].1 = "1".into();
        other[5].1 = "different physical ring".into();
        let projected = target_projection(other, &[0], true).unwrap();
        assert_eq!(logical, projected.0);
        assert_ne!(physical, projected.1);
        for key in ["gdn[L0].recurrent", "gdn[L0].conv", "gdn[L0].ef_residual"] {
            let mut changed = state.clone();
            changed.iter_mut().find(|(name, _)| name == key).unwrap().1 = "CORRUPTED".into();
            assert_ne!(
                logical,
                target_projection(changed, &[0], true).unwrap().0,
                "{key}"
            );
            let missing = state
                .iter()
                .filter(|(name, _)| name != key)
                .cloned()
                .collect();
            assert!(target_projection(missing, &[0], true).is_err(), "{key}");
        }
        let mut duplicate = state;
        duplicate.push(("gdn[L0].recurrent".into(), "duplicate".into()));
        assert!(target_projection(duplicate, &[0], true).is_err());
    }
}
