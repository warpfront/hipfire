// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Original-id routed-expert membership. Missing metadata means no pruning.
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

pub const EXPERT_KEEP_METADATA_KEY: &str = "qwen4.expert_keep";
pub const EXPERT_KEEP_SCHEMA: &str = "hipfire.qwen4.expert_keep.v1";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertKeep {
    n_experts: usize,
    layers: Vec<Vec<u16>>,
    mtp: Option<Vec<u16>>,
}

impl ExpertKeep {
    pub fn new(
        n_experts: usize,
        layers: Vec<Vec<u16>>,
        mtp: Option<Vec<u16>>,
        min_per_layer: usize,
    ) -> Result<Self, String> {
        if n_experts == 0 || n_experts > u16::MAX as usize + 1 || layers.is_empty() {
            return Err("expert keep requires nonempty layers and a u16 expert id space".into());
        }
        for (i, row) in layers
            .iter()
            .enumerate()
            .chain(mtp.iter().map(|row| (layers.len(), row)))
        {
            if row.len() < min_per_layer || row.is_empty() {
                return Err(format!(
                    "expert keep layer {i} has {} entries, minimum {min_per_layer}",
                    row.len()
                ));
            }
            if row.iter().any(|&e| e as usize >= n_experts) || row.windows(2).any(|w| w[0] >= w[1])
            {
                return Err(format!(
                    "expert keep layer {i} ids must be sorted, unique, and below {n_experts}"
                ));
            }
        }
        Ok(Self {
            n_experts,
            layers,
            mtp,
        })
    }

    pub fn from_metadata(metadata: &Value) -> Result<Option<Self>, String> {
        let Some(v) = metadata.get(EXPERT_KEEP_METADATA_KEY) else {
            return Ok(None);
        };
        if v.get("schema").and_then(Value::as_str) != Some(EXPERT_KEEP_SCHEMA) {
            return Err("unsupported expert keep schema".into());
        }
        let n_experts = integer(v, "n_experts")?;
        let layers = serde_json::from_value(
            v.get("layers")
                .cloned()
                .ok_or("missing expert keep layers")?,
        )
        .map_err(|e| format!("expert keep layers: {e}"))?;
        let mtp = serde_json::from_value(v.get("mtp").cloned().ok_or("missing expert keep mtp")?)
            .map_err(|e| format!("expert keep mtp: {e}"))?;
        let keep = Self::new(n_experts, layers, mtp, 1)?;
        if v.get("sha256").and_then(Value::as_str) != Some(keep.sha256_hex().as_str()) {
            return Err("expert keep sha256 mismatch".into());
        }
        Ok(Some(keep))
    }

    pub fn to_metadata_value(&self) -> Value {
        json!({"schema": EXPERT_KEEP_SCHEMA, "n_experts": self.n_experts, "layers": self.layers, "mtp": self.mtp, "sha256": self.sha256_hex()})
    }

    pub fn from_coder_keep_file(v: &Value) -> Result<Self, String> {
        if v.get("schema").and_then(Value::as_str) != Some("hipfire.coder.keep.v1") {
            return Err("unsupported coder keep schema".into());
        }
        let layers: Vec<Vec<u16>> =
            serde_json::from_value(v.get("keep").cloned().ok_or("missing coder keep")?)
                .map_err(|e| format!("coder keep: {e}"))?;
        if integer(v, "n_layers")? != layers.len() {
            return Err("coder keep n_layers mismatch".into());
        }
        let counts: Vec<usize> =
            serde_json::from_value(v.get("counts").cloned().ok_or("missing coder counts")?)
                .map_err(|e| format!("coder counts: {e}"))?;
        if counts != layers.iter().map(Vec::len).collect::<Vec<_>>() {
            return Err("coder keep counts mismatch".into());
        }
        let mtp: Option<Vec<u16>> = match v.get("mtp") {
            None | Some(Value::Null) => None,
            Some(m) => Some(
                serde_json::from_value(m.clone()).map_err(|e| format!("coder keep mtp: {e}"))?,
            ),
        };
        let keep = Self::new(integer(v, "n_experts")?, layers, mtp, 1)?;
        if integer(v, "total")? != keep.total_kept() {
            return Err("coder keep total mismatch".into());
        }
        Ok(keep)
    }

    pub fn is_kept(&self, layer: usize, expert: usize) -> bool {
        u16::try_from(expert).ok().is_some_and(|e| {
            self.layers
                .get(layer)
                .is_some_and(|row| row.binary_search(&e).is_ok())
        })
    }
    pub fn kept(&self, layer: usize) -> &[u16] {
        &self.layers[layer]
    }
    pub fn mtp_kept(&self) -> Option<&[u16]> {
        self.mtp.as_deref()
    }
    pub fn total_kept(&self) -> usize {
        self.layers.iter().map(Vec::len).sum()
    }
    pub fn n_experts(&self) -> usize {
        self.n_experts
    }
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }
    pub fn sha256_hex(&self) -> String {
        let canonical = serde_json::to_vec(&json!({"layers": self.layers, "mtp": self.mtp}))
            .expect("serializable keep lists");
        format!("{:x}", Sha256::digest(canonical))
    }
}

fn integer(v: &Value, key: &str) -> Result<usize, String> {
    v.get(key)
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| format!("missing or invalid {key}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_ragged_and_sha() {
        let keep = ExpertKeep::new(
            512,
            vec![(0..16).collect(), (0..512).collect()],
            Some((0..10).collect()),
            10,
        )
        .unwrap();
        let mut metadata = json!({EXPERT_KEEP_METADATA_KEY: keep.to_metadata_value()});
        assert_eq!(
            ExpertKeep::from_metadata(&metadata).unwrap(),
            Some(keep.clone())
        );
        assert_eq!(keep.total_kept(), 528);
        assert!(keep.is_kept(0, 15));
        assert!(!keep.is_kept(0, 16));
        metadata[EXPERT_KEEP_METADATA_KEY]["layers"][0][0] = json!(1);
        assert!(ExpertKeep::from_metadata(&metadata).is_err());
        metadata[EXPERT_KEEP_METADATA_KEY] = keep.to_metadata_value();
        metadata[EXPERT_KEEP_METADATA_KEY]["sha256"] = json!("bad");
        assert!(ExpertKeep::from_metadata(&metadata).is_err());
        assert_eq!(ExpertKeep::from_metadata(&json!({})).unwrap(), None);
    }

    #[test]
    fn rejects_invalid_membership() {
        for row in [vec![1, 0], vec![0, 0], vec![0, 512], vec![0]] {
            assert!(ExpertKeep::new(512, vec![row], None, 2).is_err());
        }
        assert!(ExpertKeep::new(512, vec![vec![0, 1]], Some(vec![0]), 2).is_err());
    }

    #[test]
    fn coder_keep_parse() {
        let mut v = json!({"schema":"hipfire.coder.keep.v1", "n_layers":2,"n_experts":512,"keep":[[0,3],[1,2,9]],"counts":[2,3],"total":5});
        let keep = ExpertKeep::from_coder_keep_file(&v).unwrap();
        assert_eq!(keep.kept(1), &[1, 2, 9]);
        assert_eq!(keep.mtp_kept(), None);
        v["counts"] = json!([3, 2]);
        assert!(ExpertKeep::from_coder_keep_file(&v).is_err());
        v["counts"] = json!([2, 3]);
        v["total"] = json!(6);
        assert!(ExpertKeep::from_coder_keep_file(&v).is_err());
    }

    #[test]
    fn coder_keep_parses_optional_mtp() {
        let mut v = json!({"schema":"hipfire.coder.keep.v1", "n_layers":1,"n_experts":512,"keep":[[0,3]],"counts":[2],"total":2,"mtp":null});
        assert_eq!(
            ExpertKeep::from_coder_keep_file(&v).unwrap().mtp_kept(),
            None
        );
        v["mtp"] = json!([1, 4, 7]);
        assert_eq!(
            ExpertKeep::from_coder_keep_file(&v).unwrap().mtp_kept(),
            Some(&[1u16, 4, 7][..])
        );
        v["mtp"] = json!([7, 1]);
        assert!(ExpertKeep::from_coder_keep_file(&v).is_err());
    }
}
