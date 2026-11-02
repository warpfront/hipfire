// SPDX-License-Identifier: Apache-2.0
//! Reader for the real sample directories written by `consumer24/code/extract_c24_samples.py` (c24 MQ4 hot qt44/53
//! + MQ2 cold qt133/134) and `subq4-week/artifacts/ctl3-hot/code/extract_samples.py` (W3 qt131/132 + MQ4):
//! `sample.json` {seed, layers?, provisional?, artifact_path, artifact_sha256, samples:[{layer, expert, hot,
//! tensors:[{kind, qt, shape, group, file, sha256, bytes, artifact_offset}]}]} plus raw `L{L}.E{E}.{kind}.bin`.
//! File bytes are hashed against the manifest when an expert is built (see `Scenario::build_experts`).

use crate::model::*;
use crate::scenarios::RealRef;
use serde_json::{json, Value};
use std::path::Path;

pub struct SampleSet {
    pub manifest_sha256: String,
    pub provisional: bool,
    pub artifact_sha256: Option<String>,
    pub meta: Value,
    pub gate_up: Vec<RealRef>,
    pub down: Vec<RealRef>,
}

pub fn load(dir: &Path, codec: Codec) -> Res<SampleSet> {
    let mp = dir.join("sample.json");
    let raw = std::fs::read(&mp).map_err(|e| format!("{}: {e}", mp.display()))?;
    let doc: Value = serde_json::from_slice(&raw).map_err(|e| format!("{}: {e}", mp.display()))?;
    let samples = doc
        .get("samples")
        .and_then(Value::as_array)
        .ok_or("sample.json has no `samples` array")?;
    let (mut gate_up, mut down) = (Vec::new(), Vec::new());
    for s in samples {
        let layer = s
            .get("layer")
            .and_then(Value::as_u64)
            .ok_or("sample without `layer`")?;
        let expert = s
            .get("expert")
            .and_then(Value::as_u64)
            .ok_or("sample without `expert`")?;
        let tensors = s
            .get("tensors")
            .and_then(Value::as_array)
            .ok_or("sample without `tensors`")?;
        for t in tensors {
            let kind_s = t
                .get("kind")
                .and_then(Value::as_str)
                .ok_or("tensor without `kind`")?;
            let kind = match kind_s {
                "gate_up_proj" => Kind::GateUp,
                "down_proj" => Kind::Down,
                _ => continue,
            };
            let qt = t
                .get("qt")
                .and_then(Value::as_u64)
                .ok_or("tensor without `qt`")?;
            if qt != u64::from(codec.qt(kind)) {
                continue;
            }
            let shape: Vec<u64> = t
                .get("shape")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(Value::as_u64).collect())
                .unwrap_or_default();
            if shape != [kind.m() as u64, kind.k() as u64] {
                return Err(format!(
                    "L{layer}.E{expert} {kind_s}: shape {shape:?} != [{}, {}]",
                    kind.m(),
                    kind.k()
                ));
            }
            let file = t
                .get("file")
                .and_then(Value::as_str)
                .ok_or("tensor without `file`")?;
            let sha256 = t
                .get("sha256")
                .and_then(Value::as_str)
                .ok_or("tensor without `sha256`")?
                .to_ascii_lowercase();
            let bytes = t
                .get("bytes")
                .and_then(Value::as_u64)
                .ok_or("tensor without `bytes`")? as usize;
            let r = RealRef {
                layer,
                expert,
                codec,
                path: dir.join(file),
                sha256,
                bytes,
                artifact_offset: t.get("artifact_offset").and_then(Value::as_u64),
            };
            match kind {
                Kind::GateUp => gate_up.push(r),
                Kind::Down => down.push(r),
            }
        }
    }
    let key = |r: &RealRef| (r.layer, r.expert);
    gate_up.sort_by_key(key);
    down.sort_by_key(key);
    if gate_up.is_empty() || down.is_empty() {
        return Err(format!(
            "{}: no qt{}/qt{} tensors for codec {}",
            mp.display(),
            codec.qt(Kind::GateUp),
            codec.qt(Kind::Down),
            codec.name()
        ));
    }
    let provisional = doc
        .get("provisional")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let artifact_sha256 = doc
        .get("artifact_sha256")
        .and_then(Value::as_str)
        .map(str::to_string);
    let mut meta = doc.clone();
    if let Some(o) = meta.as_object_mut() {
        o.remove("samples");
    }
    meta["selected"] = json!({"gate_up_tensors": gate_up.len(), "down_tensors": down.len(), "codec": codec.name()});
    Ok(SampleSet {
        manifest_sha256: sha256_hex(&raw),
        provisional,
        artifact_sha256,
        meta,
        gate_up,
        down,
    })
}
