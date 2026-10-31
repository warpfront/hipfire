// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Row-exactness census (oracle only): where does a verify block stop being
//! bit-identical to K sequential decode steps?
//!
//! [`RaonBundle::census_rows`] runs, from the same snapshot,
//!
//! 1. K eager decode steps (`forward_token_tapped`, one row each) with every stage
//!    tapped, then
//! 2. the verify block through the real `SpecTarget::verify_block_logits` path
//!    (eager forward with the rollback ring armed exactly as the product verify arms
//!    it, `readout_logits` head) with the same taps,
//!
//! and compares, per layer and stage (embedding, per-site mHC controls, aggregate,
//! RDA projections / conv / prepare / recurrence / gated, MLA projections / norm /
//! gated attention, mixer output, FFN input / output, site postmix streams, head
//! norm), row `r` of the verify tap with the single row of decode step `r`,
//! bitwise. The report lists, per row, the first (layer, stage) in launch order whose
//! bits differ. Taps download, so this never runs under graph capture.

use std::sync::{Arc, Mutex};

use hipfire_runtime::spec::{SpecScratch, SpecTarget};
use rdna_compute::{Gpu, GpuTensor};

use crate::bundle::RaonBundle;
use crate::forward::{forward_token_tapped, TapSite};

/// Observer installed for the verify forward (see `RaonSpecState::census_tap`).
pub type CensusTap =
    Box<dyn FnMut(&mut Gpu, TapSite, &GpuTensor, usize) -> Result<(), String> + Send>;

/// One tapped buffer: `(layer, stage name)` and its host copy (`rows * width` F32).
#[derive(Clone, Debug)]
pub struct CensusRecord {
    pub layer: usize,
    pub name: String,
    pub data: Vec<f32>,
}

/// Stage key of a tap site: `(layer, name)`; the head uses layer `n_layers`.
fn site_key(site: TapSite) -> (usize, String) {
    match site {
        TapSite::Embedding => (0, "embed".to_string()),
        TapSite::Attn(l) => (l, "attn.streams".to_string()),
        TapSite::Ffn(l) => (l, "ffn.streams".to_string()),
        TapSite::Stage(l, n) => (l, n.to_string()),
    }
}

fn download(gpu: &mut Gpu, t: &GpuTensor) -> Result<Vec<f32>, String> {
    gpu.hip
        .device_synchronize()
        .map_err(|e| format!("census: synchronize: {e:?}"))?;
    gpu.download_f32(t)
        .map_err(|e| format!("census: download: {e:?}"))
}

/// One compared stage: bit equality of every verify row against its decode row.
#[derive(Clone, Debug)]
pub struct CensusStage {
    pub layer: usize,
    pub name: String,
    /// Elements per row.
    pub width: usize,
    pub row_equal: Vec<bool>,
    pub row_max_abs: Vec<f32>,
    /// First differing element of each row (`None` = equal).
    pub row_first_diff: Vec<Option<usize>>,
}

/// Result of a census over `k` rows.
#[derive(Clone, Debug)]
pub struct CensusReport {
    pub k: usize,
    /// Stages in launch order (embedding, then per layer attention site then FFN
    /// site, then the head).
    pub stages: Vec<CensusStage>,
    /// Verify logits row r equal to decode step r's logits (bitwise).
    pub logits_equal: Vec<bool>,
    pub logits_max_abs: Vec<f32>,
    /// Decode step r's eager logits (for the caller to compare with the graph route).
    pub decode_logits: Vec<Vec<f32>>,
}

impl CensusReport {
    /// First stage (launch order) whose bits differ for row `r`.
    pub fn first_divergent(&self, r: usize) -> Option<&CensusStage> {
        self.stages.iter().find(|s| !s.row_equal[r])
    }

    /// Stages with at least one differing row.
    pub fn divergent_stages(&self) -> impl Iterator<Item = &CensusStage> {
        self.stages
            .iter()
            .filter(|s| s.row_equal.iter().any(|e| !e))
    }
}

fn compare_rows(k: usize, verify: &CensusRecord, decode: &[&CensusRecord]) -> Option<CensusStage> {
    let width = verify.data.len() / k;
    if width == 0 || verify.data.len() != width * k || decode.iter().any(|d| d.data.len() != width)
    {
        return None;
    }
    let mut row_equal = Vec::with_capacity(k);
    let mut row_max_abs = Vec::with_capacity(k);
    let mut row_first_diff = Vec::with_capacity(k);
    for (r, d) in decode.iter().enumerate() {
        let v = &verify.data[r * width..(r + 1) * width];
        let mut first = None;
        let mut max_abs = 0f32;
        for (i, (a, b)) in v.iter().zip(&d.data).enumerate() {
            if a.to_bits() != b.to_bits() {
                first.get_or_insert(i);
                let diff = (a - b).abs();
                if diff.is_nan() || diff > max_abs {
                    max_abs = diff;
                }
            }
        }
        row_equal.push(first.is_none());
        row_max_abs.push(max_abs);
        row_first_diff.push(first);
    }
    Some(CensusStage {
        layer: verify.layer,
        name: verify.name.clone(),
        width,
        row_equal,
        row_max_abs,
        row_first_diff,
    })
}

impl RaonBundle {
    /// Census of a verify block over `chain` (K = `chain.len()` rows) against K
    /// sequential eager decode steps. The slot must sit at the prompt position;
    /// it is restored to it before returning (also on a verify error after the
    /// snapshot). `spec` is the bundle's speculation scratch
    /// (`SpecTarget::new_spec_scratch`).
    pub fn census_rows(
        &mut self,
        gpu: &mut Gpu,
        chain: &[u32],
        spec: &mut dyn SpecScratch,
    ) -> Result<CensusReport, String> {
        let k = chain.len();
        if k == 0 {
            return Err("census: empty chain".into());
        }
        let position = self.state.n_tokens();
        let mut snap = self.state.snapshot_alloc(gpu)?;
        self.state.snapshot_into(gpu, &mut snap)?;
        let n_layers = self.config.n_layers;

        // 1. K sequential eager decode steps, every stage tapped.
        let mut decode: Vec<Vec<CensusRecord>> = Vec::with_capacity(k);
        let mut decode_logits = Vec::with_capacity(k);
        let seq = (|| -> Result<(), String> {
            for (i, &token) in chain.iter().enumerate() {
                let mut records: Vec<CensusRecord> = Vec::new();
                {
                    let mut tap = |gpu: &mut Gpu,
                                   site: TapSite,
                                   t: &GpuTensor,
                                   _rows: usize|
                     -> Result<(), String> {
                        let (layer, name) = site_key(site);
                        records.push(CensusRecord {
                            layer,
                            name,
                            data: download(gpu, t)?,
                        });
                        Ok(())
                    };
                    forward_token_tapped(
                        &self.config,
                        &self.weights,
                        &mut self.state,
                        &mut self.scratch,
                        &mut self.work,
                        gpu,
                        token,
                        (position + i) as u32,
                        true,
                        Some(&mut tap),
                    )?;
                }
                records.push(CensusRecord {
                    layer: n_layers,
                    name: "head.norm".to_string(),
                    data: download(gpu, &self.scratch.final_norm)?,
                });
                decode_logits.push(download(gpu, &self.scratch.logits)?);
                decode.push(records);
            }
            Ok(())
        })();
        let restored_seq = self.state.restore(gpu, &snap);
        seq?;
        restored_seq?;
        if self.state.n_tokens() != position {
            return Err(format!(
                "census: restore left n_tokens {} != {position}",
                self.state.n_tokens()
            ));
        }

        // 2. The verify block through the product path, same taps.
        let sink: Arc<Mutex<Vec<CensusRecord>>> = Arc::new(Mutex::new(Vec::new()));
        let sink_tap = Arc::clone(&sink);
        self.spec.census_tap = Some(Box::new(move |gpu, site, t, _rows| {
            let (layer, name) = site_key(site);
            sink_tap.lock().expect("census sink").push(CensusRecord {
                layer,
                name,
                data: download(gpu, t)?,
            });
            Ok(())
        }));
        let verified = SpecTarget::verify_block_logits(self, gpu, chain, position, spec, None);
        self.spec.census_tap = None;
        let abandoned = SpecTarget::abandon_verify(self, gpu, spec);
        let freed = snap.free_gpu(gpu);
        let logits = verified?;
        abandoned?;
        freed?;
        if self.state.n_tokens() != position {
            return Err(format!(
                "census: abandon_verify left n_tokens {} != {position}",
                self.state.n_tokens()
            ));
        }
        let verify = sink.lock().expect("census sink");

        let vocab = self.config.vocab;
        if logits.len() != k * vocab {
            return Err(format!(
                "census: verify returned {} logits for K={k}",
                logits.len()
            ));
        }
        let mut logits_equal = Vec::with_capacity(k);
        let mut logits_max_abs = Vec::with_capacity(k);
        for r in 0..k {
            let v = &logits[r * vocab..(r + 1) * vocab];
            let d = &decode_logits[r];
            logits_equal.push(v.iter().zip(d).all(|(a, b)| a.to_bits() == b.to_bits()));
            logits_max_abs.push(
                v.iter()
                    .zip(d)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, |m, x| if x.is_nan() || x > m { x } else { m }),
            );
        }

        let mut stages = Vec::with_capacity(verify.len());
        for rec in verify.iter() {
            let rows: Option<Vec<&CensusRecord>> = decode
                .iter()
                .map(|step| {
                    step.iter()
                        .find(|d| d.layer == rec.layer && d.name == rec.name)
                })
                .collect();
            let Some(rows) = rows else { continue };
            if let Some(stage) = compare_rows(k, rec, &rows) {
                stages.push(stage);
            }
        }
        Ok(CensusReport {
            k,
            stages,
            logits_equal,
            logits_max_abs,
            decode_logits,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(name: &str, data: Vec<f32>) -> CensusRecord {
        CensusRecord {
            layer: 3,
            name: name.to_string(),
            data,
        }
    }

    #[test]
    fn compare_rows_is_bitwise_and_row_wise() {
        let verify = rec("x", vec![1.0, 2.0, 3.0, f32::from_bits(0x7fc0_0001)]);
        let r0 = rec("x", vec![1.0, 2.0]);
        let r1 = rec("x", vec![3.0, f32::from_bits(0x7fc0_0002)]);
        let stage = compare_rows(2, &verify, &[&r0, &r1]).expect("widths agree");
        assert_eq!(stage.width, 2);
        // Row 0 is bit-equal; row 1's NaN payload differs, which a float `==` would hide.
        assert_eq!(stage.row_equal, vec![true, false]);
        assert_eq!(stage.row_first_diff, vec![None, Some(1)]);
        // -0.0 vs +0.0 differ bitwise.
        let z = compare_rows(1, &rec("x", vec![-0.0]), &[&rec("x", vec![0.0])]).unwrap();
        assert_eq!(z.row_equal, vec![false]);
    }

    #[test]
    fn compare_rows_refuses_mismatched_widths() {
        let verify = rec("x", vec![1.0; 4]);
        let d = rec("x", vec![1.0; 3]);
        assert!(compare_rows(2, &verify, &[&d, &d]).is_none());
        assert!(compare_rows(3, &verify, &[&d, &d, &d]).is_none());
    }

    #[test]
    fn site_keys_name_every_tap_site() {
        assert_eq!(site_key(TapSite::Embedding), (0, "embed".to_string()));
        assert_eq!(site_key(TapSite::Attn(4)), (4, "attn.streams".to_string()));
        assert_eq!(site_key(TapSite::Ffn(4)), (4, "ffn.streams".to_string()));
        assert_eq!(
            site_key(TapSite::Stage(7, "rda.q")),
            (7, "rda.q".to_string())
        );
    }
}
