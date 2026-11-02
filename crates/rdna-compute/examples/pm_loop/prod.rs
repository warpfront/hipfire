// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! A generic `Workload` for the PM producer generators (A4/C8 rotate and
//! RMSNorm producers, the MoE down rotate, the GDN chunk-prep preamble, ...):
//! one launch over read-only inputs and a few output buffers, no accumulating
//! epilogue. A producer module implements only [`Producer`] (knobs, shape
//! presets, the two objects, deterministic inputs, kernarg packing, optional
//! exact CPU reference and emulator run); [`ProducerWorkload`] does the device
//! buffers, per-arm outputs, restore-before-launch, comparison and the
//! baseline-identity criterion.
//!
//! Pass criterion (`requires_baseline_identity`): the candidate's compared
//! outputs are bitwise equal to the shipped object's, and, where the producer
//! has an exact CPU reference, equal to it too.
//!
//! Included with `#[path = "pm_loop/prod.rs"] mod prod;`.
#![allow(dead_code)]

use crate::common::{dev_err, max_diff, Elem};
use crate::workload::{Arm, Launch, Object, RefReport, Result, Workload};
use rdna_compute::Gpu;
use serde_json::{json, Value};
use std::cell::OnceCell;
use std::ops::Range;

/// One output buffer of the producer's launch.
#[derive(Clone, Debug)]
pub struct OutBuf {
    pub name: &'static str,
    /// Bytes the buffer holds before every launch (zeros for a pure output;
    /// the input state for an in-place one such as a conv ring).
    pub init: Vec<u8>,
    pub elem: Elem,
    /// Part of the correctness payload (compared to the baseline and the
    /// reference). Non-compared buffers are scratch.
    pub compare: bool,
}

impl OutBuf {
    pub fn zeros(name: &'static str, bytes: usize, elem: Elem) -> Self {
        Self {
            name,
            init: vec![0; bytes],
            elem,
            compare: true,
        }
    }
}

/// Device addresses of one arm's buffers, in the order of
/// [`Producer::inputs`] / [`Producer::outputs`].
#[derive(Clone, Debug)]
pub struct Ptrs {
    pub inputs: Vec<u64>,
    pub outputs: Vec<u64>,
}

/// Expected bytes of one output buffer: `bytes` has the buffer's full length
/// and is meaningful only on `ranges` (a sampled reference covers a subset).
#[derive(Clone, Debug)]
pub struct Covered {
    pub bytes: Vec<u8>,
    pub ranges: Vec<Range<usize>>,
}

impl Covered {
    pub fn full(bytes: Vec<u8>) -> Self {
        let n = bytes.len();
        Self {
            bytes,
            ranges: vec![0..n],
        }
    }
    pub fn covered_bytes(&self) -> usize {
        self.ranges.iter().map(|r| r.len()).sum()
    }
}

/// An exact CPU reference's expectation: one entry per output buffer (`None`
/// = that buffer has no reference), in [`Producer::outputs`] order.
#[derive(Clone, Debug)]
pub struct Expect {
    pub outputs: Vec<Option<Covered>>,
    /// Name of the reference (e.g. `cpu_fwht128_i4_c2`), recorded verbatim.
    pub reference: String,
}

/// One PM producer kernel under test.
pub trait Producer {
    /// `--kernel` name.
    fn kernel(&self) -> &'static str;
    fn shape_json(&self) -> Value;
    /// Resolved knobs (the export, the Spec axes).
    fn knobs_json(&self) -> Value;
    fn record_extra(&self) -> Vec<(String, Value)> {
        Vec::new()
    }
    /// Emit + link the candidate (native writer, the `peacemaker-isa emit --co` path).
    fn candidate(&self) -> Result<Object>;
    /// The shipped object and the symbol production launches.
    fn baseline(&self) -> Result<Object>;
    /// Deterministic read-only inputs, uploaded once and shared by both arms.
    fn inputs(&self) -> Vec<Vec<u8>>;
    /// Output buffers (each arm gets its own copy, restored before every launch).
    fn outputs(&self) -> Vec<OutBuf>;
    /// The launch exactly as the production launcher issues it.
    fn launch(&self, p: &Ptrs) -> Launch;
    /// Exact CPU reference over `inputs()`; `Err(reason)` = the repo has none
    /// for this shape/export (the record then reports the reason and the pass
    /// criterion is baseline identity alone).
    fn reference(&self, inputs: &[Vec<u8>], outputs: &[OutBuf]) -> Result<Expect>;
    /// `--emu`: run the candidate on the emulator at a small shape (see
    /// `common::emu_run`) or `common::refused(reason)`. Never fabricate.
    fn emu(&self) -> Value;
}

struct Dev {
    inputs: Vec<hip_bridge::DeviceBuffer>,
    /// Per arm: one buffer per output.
    outs: [Vec<hip_bridge::DeviceBuffer>; 2],
}

fn idx(arm: Arm) -> usize {
    match arm {
        Arm::Baseline => 0,
        Arm::Candidate => 1,
    }
}

pub struct ProducerWorkload<P: Producer> {
    p: P,
    outs: Vec<OutBuf>,
    host_inputs: Option<Vec<Vec<u8>>>,
    dev: Option<Dev>,
    expect: OnceCell<Result<Expect>>,
}

/// Box a producer as a driver workload.
pub fn boxed<P: Producer + 'static>(p: P) -> Box<dyn Workload> {
    let outs = p.outputs();
    Box::new(ProducerWorkload {
        p,
        outs,
        host_inputs: None,
        dev: None,
        expect: OnceCell::new(),
    })
}

impl<P: Producer> ProducerWorkload<P> {
    fn compared(&self) -> impl Iterator<Item = (usize, &OutBuf)> {
        self.outs.iter().enumerate().filter(|(_, o)| o.compare)
    }
    fn payload_bytes(&self) -> usize {
        self.compared().map(|(_, o)| o.init.len()).sum()
    }
    fn ptrs(&self, arm: Arm) -> Ptrs {
        let d = self.dev.as_ref().expect("launch before setup");
        let ptr = |b: &hip_bridge::DeviceBuffer| b.as_ptr() as u64;
        Ptrs {
            inputs: d.inputs.iter().map(ptr).collect(),
            outputs: d.outs[idx(arm)].iter().map(ptr).collect(),
        }
    }
}

impl<P: Producer> Workload for ProducerWorkload<P> {
    fn kernel(&self) -> &'static str {
        self.p.kernel()
    }
    fn arch(&self) -> &'static str {
        "gfx1201"
    }
    fn shape_json(&self) -> Value {
        self.p.shape_json()
    }
    fn candidate_knobs(&self) -> Value {
        self.p.knobs_json()
    }
    fn record_extra(&self) -> Vec<(String, Value)> {
        self.p.record_extra()
    }
    fn candidate(&self) -> Result<Object> {
        self.p.candidate()
    }
    fn baseline(&self) -> Result<Object> {
        self.p.baseline()
    }

    fn setup(&mut self, gpu: &Gpu) -> Result<()> {
        let host = self.p.inputs();
        let up = |b: &[u8]| -> Result<hip_bridge::DeviceBuffer> {
            let d = gpu.hip.malloc(b.len().max(16)).map_err(dev_err)?;
            if !b.is_empty() {
                gpu.hip.memcpy_htod(&d, b).map_err(dev_err)?;
            }
            Ok(d)
        };
        let inputs = host.iter().map(|b| up(b)).collect::<Result<Vec<_>>>()?;
        let mut outs = [Vec::new(), Vec::new()];
        for arm_outs in &mut outs {
            for o in &self.outs {
                arm_outs.push(up(&o.init)?);
            }
        }
        self.host_inputs = Some(host);
        self.dev = Some(Dev { inputs, outs });
        Ok(())
    }

    fn launch(&self, arm: Arm) -> Launch {
        self.p.launch(&self.ptrs(arm))
    }

    fn reset(&self, gpu: &Gpu, arm: Arm) -> Result<()> {
        let d = self.dev.as_ref().ok_or("reset before setup")?;
        for (o, buf) in self.outs.iter().zip(&d.outs[idx(arm)]) {
            gpu.hip.memcpy_htod(buf, &o.init).map_err(dev_err)?;
        }
        Ok(())
    }

    fn output(&self, gpu: &Gpu, arm: Arm) -> Result<Vec<u8>> {
        let d = self.dev.as_ref().ok_or("output before setup")?;
        let mut all = Vec::with_capacity(self.payload_bytes());
        for (i, o) in self.compared() {
            let buf = &d.outs[idx(arm)][i];
            let mut v = vec![0u8; buf.size()];
            gpu.hip.memcpy_dtoh(&mut v, buf).map_err(dev_err)?;
            v.truncate(o.init.len());
            all.extend_from_slice(&v);
        }
        Ok(all)
    }

    fn check(&self, out: &[u8]) -> Result<RefReport> {
        if out.len() != self.payload_bytes() {
            return Err(format!(
                "output is {} bytes, expected {}",
                out.len(),
                self.payload_bytes()
            ));
        }
        let host = self.host_inputs.as_ref().ok_or("check before setup")?;
        let expect = self
            .expect
            .get_or_init(|| self.p.reference(host, &self.outs))
            .as_ref()
            .map_err(|e| e.clone())?;
        let (mut mismatching, mut covered, mut worst, mut off) = (0usize, 0usize, 0f64, 0usize);
        let mut per_buf = Vec::new();
        for (i, o) in self.compared() {
            let got = &out[off..off + o.init.len()];
            off += o.init.len();
            let Some(Some(c)) = expect.outputs.get(i) else {
                per_buf.push(format!("{}:uncovered", o.name));
                continue;
            };
            if c.bytes.len() != got.len() {
                return Err(format!(
                    "reference for {} is {} bytes, output {}",
                    o.name,
                    c.bytes.len(),
                    got.len()
                ));
            }
            let mut bad = 0usize;
            for r in &c.ranges {
                let (g, e) = (&got[r.clone()], &c.bytes[r.clone()]);
                if g != e {
                    let w = o.elem.width();
                    bad += g.chunks(w).zip(e.chunks(w)).filter(|(x, y)| x != y).count();
                    worst = worst.max(max_diff(o.elem, g, e));
                }
            }
            mismatching += bad;
            covered += c.covered_bytes();
            per_buf.push(format!(
                "{}:{}/{}B,{}bad",
                o.name,
                c.covered_bytes(),
                got.len(),
                bad
            ));
        }
        if covered == 0 {
            return Err(format!(
                "{}: the reference covers no output byte",
                expect.reference
            ));
        }
        Ok(RefReport {
            reference: format!(
                "{}({};mismatching_elements={mismatching})",
                expect.reference,
                per_buf.join(",")
            ),
            max_abs_err: worst,
            tolerance: 0.0,
            pass: mismatching == 0,
        })
    }

    fn max_abs_diff(&self, a: &[u8], b: &[u8]) -> Result<f64> {
        if a.len() != b.len() || a.len() != self.payload_bytes() {
            return Err(format!(
                "length mismatch {} vs {} (payload {})",
                a.len(),
                b.len(),
                self.payload_bytes()
            ));
        }
        let mut off = 0;
        let mut worst = 0f64;
        for (_, o) in self.compared() {
            let n = o.init.len();
            worst = worst.max(max_diff(o.elem, &a[off..off + n], &b[off..off + n]));
            off += n;
        }
        Ok(worst)
    }

    fn requires_baseline_identity(&self) -> bool {
        true
    }

    fn emu(&self) -> Value {
        self.p.emu()
    }
}

/// The `shape` JSON of a producer: its dimensions plus the byte sizes of
/// every input and output buffer.
pub fn buffers_json(inputs: &[(&str, usize)], outs: &[OutBuf]) -> Value {
    json!({
        "inputs": inputs.iter().map(|(n, b)| json!({"name": n, "bytes": b})).collect::<Vec<_>>(),
        "outputs": outs.iter().map(|o| json!({"name": o.name, "bytes": o.init.len(), "compared": o.compare})).collect::<Vec<_>>(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A producer with no GPU: `y[i] = 2 * x[i]` over f32.
    struct Doubler {
        n: usize,
        reference: bool,
    }

    impl Producer for Doubler {
        fn kernel(&self) -> &'static str {
            "doubler"
        }
        fn shape_json(&self) -> Value {
            json!({"n": self.n})
        }
        fn knobs_json(&self) -> Value {
            json!({})
        }
        fn candidate(&self) -> Result<Object> {
            Err("no candidate".into())
        }
        fn baseline(&self) -> Result<Object> {
            Err("no baseline".into())
        }
        fn inputs(&self) -> Vec<Vec<u8>> {
            vec![crate::common::f32_unit_bytes(1, self.n, 1.0)]
        }
        fn outputs(&self) -> Vec<OutBuf> {
            vec![
                OutBuf::zeros("y", 4 * self.n, Elem::F32),
                OutBuf {
                    compare: false,
                    ..OutBuf::zeros("scratch", 64, Elem::U8)
                },
            ]
        }
        fn launch(&self, p: &Ptrs) -> Launch {
            let mut k = Vec::new();
            crate::common::put_ptr(&mut k, p.inputs[0]);
            crate::common::put_ptr(&mut k, p.outputs[0]);
            Launch {
                kernarg: k,
                grid: [1, 1, 1],
                block: [32, 1, 1],
                lds: 0,
            }
        }
        fn reference(&self, inputs: &[Vec<u8>], outputs: &[OutBuf]) -> Result<Expect> {
            if !self.reference {
                return Err("no CPU reference for doubler".into());
            }
            let y: Vec<u8> = inputs[0]
                .chunks_exact(4)
                .flat_map(|c| (f32::from_le_bytes(c.try_into().unwrap()) * 2.0).to_le_bytes())
                .collect();
            assert_eq!(y.len(), outputs[0].init.len());
            Ok(Expect {
                outputs: vec![Some(Covered::full(y)), None],
                reference: "cpu_double".into(),
            })
        }
        fn emu(&self) -> Value {
            crate::common::refused("test")
        }
    }

    fn wl(reference: bool) -> ProducerWorkload<Doubler> {
        let p = Doubler { n: 8, reference };
        let outs = p.outputs();
        let host = p.inputs();
        ProducerWorkload {
            p,
            outs,
            host_inputs: Some(host),
            dev: None,
            expect: OnceCell::new(),
        }
    }

    fn doubled(w: &ProducerWorkload<Doubler>) -> Vec<u8> {
        w.host_inputs.as_ref().unwrap()[0]
            .chunks_exact(4)
            .flat_map(|c| (f32::from_le_bytes(c.try_into().unwrap()) * 2.0).to_le_bytes())
            .collect()
    }

    #[test]
    fn payload_is_the_compared_outputs_only() {
        let w = wl(true);
        assert_eq!(
            w.payload_bytes(),
            32,
            "the 64 B scratch buffer is not compared"
        );
        assert!(w.requires_baseline_identity());
        let j = buffers_json(&[("x", 32)], &w.outs);
        assert_eq!(j["outputs"][1]["compared"], false);
    }

    #[test]
    fn check_passes_exact_and_counts_mismatches() {
        let w = wl(true);
        let good = doubled(&w);
        let r = w.check(&good).unwrap();
        assert!(r.pass && r.max_abs_err == 0.0 && r.tolerance == 0.0);
        assert!(
            r.reference
                .starts_with("cpu_double(y:32/32B,0bad;mismatching_elements=0)"),
            "{}",
            r.reference
        );
        let mut bad = good.clone();
        bad[4..8].copy_from_slice(&123.0f32.to_le_bytes());
        let r = wl(true).check(&bad).unwrap();
        assert!(!r.pass && r.max_abs_err > 100.0);
        assert!(r.reference.contains("1bad"), "{}", r.reference);
        assert!(wl(true).check(&good[..28]).is_err());
    }

    #[test]
    fn sampled_reference_only_judges_its_ranges() {
        struct Sampled;
        impl Producer for Sampled {
            fn kernel(&self) -> &'static str {
                "s"
            }
            fn shape_json(&self) -> Value {
                json!({})
            }
            fn knobs_json(&self) -> Value {
                json!({})
            }
            fn candidate(&self) -> Result<Object> {
                Err(String::new())
            }
            fn baseline(&self) -> Result<Object> {
                Err(String::new())
            }
            fn inputs(&self) -> Vec<Vec<u8>> {
                vec![]
            }
            fn outputs(&self) -> Vec<OutBuf> {
                vec![OutBuf::zeros("y", 16, Elem::U8)]
            }
            fn launch(&self, _: &Ptrs) -> Launch {
                unreachable!()
            }
            fn reference(&self, _: &[Vec<u8>], _: &[OutBuf]) -> Result<Expect> {
                Ok(Expect {
                    outputs: vec![Some(Covered {
                        bytes: vec![7; 16],
                        ranges: vec![0..4, 12..16],
                    })],
                    reference: "cpu_sampled".into(),
                })
            }
            fn emu(&self) -> Value {
                Value::Null
            }
        }
        let mk = || ProducerWorkload {
            p: Sampled,
            outs: Sampled.outputs(),
            host_inputs: Some(vec![]),
            dev: None,
            expect: OnceCell::new(),
        };
        // Bytes outside the covered ranges are not judged.
        let mut got = vec![7u8; 16];
        got[8] = 0;
        let r = mk().check(&got).unwrap();
        assert!(r.pass, "{}", r.reference);
        assert!(r.reference.contains("y:8/16B"), "{}", r.reference);
        got[13] = 9;
        assert!(!mk().check(&got).unwrap().pass);
    }

    #[test]
    fn missing_reference_is_an_error_with_the_reason() {
        let w = wl(false);
        let e = w.check(&doubled(&w)).unwrap_err();
        assert_eq!(e, "no CPU reference for doubler");
    }

    #[test]
    fn baseline_diff_is_per_buffer_dtype() {
        let w = wl(true);
        let a = doubled(&w);
        let mut b = a.clone();
        assert_eq!(w.max_abs_diff(&a, &b).unwrap(), 0.0);
        b[0..4].copy_from_slice(
            &(f32::from_le_bytes(a[0..4].try_into().unwrap()) + 0.5).to_le_bytes(),
        );
        assert_eq!(w.max_abs_diff(&a, &b).unwrap(), 0.5);
        assert!(w.max_abs_diff(&a, &b[..28]).is_err());
    }
}
