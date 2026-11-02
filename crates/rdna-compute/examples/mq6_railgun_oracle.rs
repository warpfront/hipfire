// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Explicit prepared railgun MQ6 parity and retained-program reuse oracle.
//! Independent original HIP singleton arithmetic; complete output guards and
//! inactive columns; same-address A/B/A; safe wrong-X and corruption controls.
//! No model, automatic capture/decode routing, timing or quality claim.

use hip_bridge::KernargBlob;
use railgun_corpus::JitCorpus;
use rdna_compute::mq6_railgun::{Mq6Invocation, Mq6TensorPart, PreparedMq6Program};
use rdna_compute::{kernels, DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

type Result<T> = std::result::Result<T, String>;
const MAX_B: usize = 8;
const GUARD: usize = 64;
const POISON: u8 = 0x7f;
const AMPS: [f32; MAX_B] = [1.0, 1.0 / 1024.0, 1.0 / 16.0, 4.0, 64.0, 0.5, 0.0, 1024.0];
const HIP_ANCHOR: &str = "gemv_mq6g256v2";
const PM_SINGLE: &str = "gemv_mq6g256v2_xbatch_pm";
const PM_X4: &str = "gemv_mq6g256v2_x4_xbatch_pm";

fn err(value: impl std::fmt::Display) -> String {
    value.to_string()
}
fn check(value: bool, why: impl Into<String>) -> Result<()> {
    if value {
        Ok(())
    } else {
        Err(why.into())
    }
}
fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .collect()
}

struct Evidence(BufWriter<File>);
impl Evidence {
    fn emit(&mut self, value: Value) -> Result<()> {
        serde_json::to_writer(&mut self.0, &value).map_err(err)?;
        self.0.write_all(b"\n").map_err(err)?;
        self.0.flush().map_err(err)?;
        println!("{value}");
        Ok(())
    }
    /// Never unwind through borrowed allocation owners after an unexpected
    /// runtime error: its quiescence may be Unknown. The root wrapper keeps
    /// this process's failure output and performs any device recovery.
    fn runtime_failure(&mut self, stage: &str, failure: impl std::fmt::Debug) -> ! {
        let message = format!("{failure:?}");
        let _ = self.emit(json!({"kind":"runtime_failure","stage":stage,
            "error":message,"passed":false,"destructors_skipped":true}));
        eprintln!("mq6_railgun_oracle {stage}: {message}");
        std::process::exit(1)
    }
}

// Same seeded finite MQ6/X conventions as the earlier native oracle.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn below(&mut self, limit: u64) -> u64 {
        self.next() % limit
    }
    fn fill(&mut self, out: &mut [u8]) {
        let mut chunks = out.chunks_exact_mut(8);
        for chunk in &mut chunks {
            chunk.copy_from_slice(&self.next().to_le_bytes());
        }
        let rest = chunks.into_remainder();
        let tail = self.next().to_le_bytes();
        rest.copy_from_slice(&tail[..rest.len()]);
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
    fn f16(&mut self, lo: u16, hi: u16) -> u16 {
        let sign = (self.below(2) as u16) << 15;
        match self.below(32) {
            0 => sign,
            1 => sign | (1 + self.below(0x3ff) as u16),
            _ => {
                sign | ((15 - lo + self.below(u64::from(lo + hi + 1)) as u16) << 10)
                    | self.below(0x400) as u16
            }
        }
    }
}
fn gen_weight(m: usize, k: usize, rng: &mut Rng) -> Vec<u8> {
    let mut bytes = vec![0u8; m * (k / 256) * 200];
    for group in bytes.chunks_exact_mut(200) {
        rng.fill(&mut group[8..]);
        for half in 0..2 {
            let scale = rng.f16(9, 2);
            let zero = rng.f16(7, 5);
            group[half * 4..half * 4 + 2].copy_from_slice(&scale.to_le_bytes());
            group[half * 4 + 2..half * 4 + 4].copy_from_slice(&zero.to_le_bytes());
        }
    }
    bytes
}
fn gen_x(k: usize, rng: &mut Rng) -> Vec<f32> {
    let mut values = Vec::with_capacity(MAX_B * k);
    for amp in AMPS {
        for _ in 0..k {
            values.push(match rng.below(64) {
                _ if amp == 0.0 => {
                    if rng.below(2) == 0 {
                        0.0
                    } else {
                        -0.0
                    }
                }
                0 => 0.0,
                1 => -0.0,
                2..=5 => f32::from_bits(((rng.below(2) as u32) << 31) | rng.below(1 << 23) as u32),
                _ => rng.unit() * amp,
            });
        }
    }
    values
}

struct Output {
    owner: GpuTensor,
    view: GpuTensor,
    m: usize,
}
impl Output {
    fn new(gpu: &mut Gpu, m: usize) -> Result<Self> {
        let owner = gpu
            .zeros(&[2 * GUARD + MAX_B * m], DType::F32)
            .map_err(err)?;
        let view = owner.sub_offset(GUARD, MAX_B * m);
        Ok(Self { owner, view, m })
    }
    fn bytes(&self) -> usize {
        (2 * GUARD + MAX_B * self.m) * 4
    }
    fn reset(&self, gpu: &Gpu) -> Result<()> {
        // hipMemset may return before its default-stream work completes.
        // PreparedMq6Program joins the current HIP producer stream, so the
        // poison fill must be ordered on that stream too. A pre-fill device
        // sync does not order this later write against the independent PM4 IB.
        match gpu.active_stream.as_ref() {
            Some(stream) => {
                gpu.hip
                    .memset_async(&self.owner.buf, i32::from(POISON), self.bytes(), stream)
            }
            None => gpu
                .hip
                .memset(&self.owner.buf, i32::from(POISON), self.bytes()),
        }
        .map_err(err)
    }
    fn read(&self, gpu: &Gpu) -> Result<Vec<u8>> {
        let mut bytes = vec![0u8; self.owner.buf.size()];
        gpu.hip
            .memcpy_dtoh(&mut bytes, &self.owner.buf)
            .map_err(err)?;
        bytes.truncate(self.bytes());
        Ok(bytes)
    }
    fn active(&self, bytes: &[u8], n: usize) -> Result<()> {
        check(
            bytes.len() == self.bytes() && (1..=MAX_B).contains(&n),
            "active bounds",
        )?;
        for (column, &amp) in AMPS.iter().take(n).enumerate() {
            let start = (GUARD + column * self.m) * 4;
            let mut nonzero = false;
            for (row, word) in bytes[start..start + self.m * 4].chunks_exact(4).enumerate() {
                check(
                    word != [POISON; 4],
                    format!("unwritten active word c={column} r={row}"),
                )?;
                let value = f32::from_le_bytes(word.try_into().unwrap());
                check(
                    value.is_finite(),
                    format!("nonfinite active word c={column} r={row}"),
                )?;
                nonzero |= value != 0.0;
            }
            check(
                self.m == 0 || amp == 0.0 || nonzero,
                format!("vacuous column {column}"),
            )?;
        }
        Ok(())
    }
    fn expected(&self, reference: &[u8], n: usize) -> Vec<u8> {
        let mut bytes = vec![POISON; self.bytes()];
        let start = GUARD * 4;
        let end = start + n * self.m * 4;
        bytes[start..end].copy_from_slice(&reference[start..end]);
        bytes
    }
    fn compare(&self, reference: &[u8], actual: &[u8], n: usize) -> Result<()> {
        self.active(reference, n)
            .map_err(|e| format!("HIP reference: {e}"))?;
        self.active(actual, n)
            .map_err(|e| format!("replay output: {e}"))?;
        let expected = self.expected(reference, n);
        if let Some(i) = expected.iter().zip(actual).position(|(a, b)| a != b) {
            return Err(format!(
                "fullbuffer mismatch byte={i} expected={:02x} actual={:02x}",
                expected[i], actual[i]
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct Shape {
    name: &'static str,
    ms: [usize; 4],
    k: usize,
    x4: bool,
    native: bool,
}
const SHAPES: [Shape; 7] = [
    Shape {
        name: "ragged_single_19",
        ms: [19, 0, 0, 0],
        k: 256,
        x4: false,
        native: false,
    },
    Shape {
        name: "ragged_single_33",
        ms: [33, 0, 0, 0],
        k: 512,
        x4: false,
        native: false,
    },
    Shape {
        name: "x4_empty_middle",
        ms: [19, 0, 7, 0],
        k: 256,
        x4: true,
        native: false,
    },
    Shape {
        name: "x4_empty_first",
        ms: [0, 1, 0, 33],
        k: 512,
        x4: true,
        native: false,
    },
    Shape {
        name: "native_gdn_qkv",
        ms: [10240, 0, 0, 0],
        k: 2560,
        x4: false,
        native: true,
    },
    Shape {
        name: "native_out",
        ms: [2560, 0, 0, 0],
        k: 6144,
        x4: false,
        native: true,
    },
    Shape {
        name: "native_qsa_x4",
        ms: [12288, 640, 512, 512],
        k: 2560,
        x4: true,
        native: true,
    },
];

struct Fixture {
    shape: Shape,
    // These are original allocation owners, never just shallow clones.
    weights: Vec<GpuTensor>,
    outputs: Vec<Output>,
    x: GpuTensor,
    a: Vec<f32>,
    b: Vec<f32>,
}
impl Fixture {
    fn new(gpu: &mut Gpu, shape: Shape, seed: u64, log: &mut Evidence) -> Result<Self> {
        let mut rng = Rng(seed);
        let mut weights = Vec::new();
        let mut outputs = Vec::new();
        let mut hashes = Vec::new();
        for &m in &shape.ms[..if shape.x4 { 4 } else { 1 }] {
            let bytes = if m == 0 {
                vec![0u8; 256]
            } else {
                gen_weight(m, shape.k, &mut rng)
            };
            hashes.push(sha(&bytes));
            weights.push(gpu.upload_raw(&bytes, &[bytes.len()]).map_err(err)?);
            outputs.push(Output::new(gpu, m)?);
        }
        let a = gen_x(shape.k, &mut rng);
        let b = gen_x(shape.k, &mut rng);
        check(f32_bytes(&a) != f32_bytes(&b), "A/B inputs identical")?;
        let x = gpu.upload_f32(&a, &[MAX_B * shape.k]).map_err(err)?;
        log.emit(
            json!({"kind":"fixture","shape":shape.name,"ms":shape.ms,"k":shape.k,
            "seed":seed,"weight_sha256":hashes,"x_a_sha256":sha(&f32_bytes(&a)),
            "x_b_sha256":sha(&f32_bytes(&b)),"x_address":format!("{:p}",x.buf.as_ptr())}),
        )?;
        Ok(Self {
            shape,
            weights,
            outputs,
            x,
            a,
            b,
        })
    }
    fn invocation<'a>(&'a self, x: &'a GpuTensor, n: usize) -> Mq6Invocation<'a> {
        let part = |i: usize| Mq6TensorPart {
            weight: &self.weights[i],
            output: &self.outputs[i].view,
            m: self.shape.ms[i] as u32,
        };
        if self.shape.x4 {
            Mq6Invocation::X4 {
                parts: std::array::from_fn(|i| (self.shape.ms[i] > 0).then(|| part(i))),
                x,
                k: self.shape.k as u32,
                batch: n as u32,
            }
        } else {
            Mq6Invocation::Single {
                part: part(0),
                x,
                k: self.shape.k as u32,
                batch: n as u32,
            }
        }
    }
    fn reset(&self, gpu: &Gpu) -> Result<()> {
        gpu.hip.device_synchronize().map_err(err)?;
        self.outputs.iter().try_for_each(|out| out.reset(gpu))
    }
    fn input(&self, gpu: &Gpu, b: bool) -> Result<()> {
        gpu.hip.device_synchronize().map_err(err)?;
        gpu.hip
            .memcpy_htod(&self.x.buf, &f32_bytes(if b { &self.b } else { &self.a }))
            .map_err(err)
    }
    fn read(&self, gpu: &Gpu) -> Result<Vec<Vec<u8>>> {
        gpu.hip.device_synchronize().map_err(err)?;
        self.outputs.iter().map(|out| out.read(gpu)).collect()
    }
    fn compare(&self, reference: &[Vec<u8>], actual: &[Vec<u8>], n: usize) -> Result<()> {
        check(
            reference.len() == self.outputs.len() && actual.len() == self.outputs.len(),
            "part count",
        )?;
        for (i, ((out, want), got)) in self.outputs.iter().zip(reference).zip(actual).enumerate() {
            out.compare(want, got, n)
                .map_err(|e| format!("{} B={n} part={i}: {e}", self.shape.name))?;
        }
        Ok(())
    }
    fn reference(&self, gpu: &Gpu) -> Result<Vec<Vec<u8>>> {
        self.reference_with_x(gpu, &self.x)
    }
    fn reference_with_x(&self, gpu: &Gpu, x: &GpuTensor) -> Result<Vec<Vec<u8>>> {
        self.reset(gpu)?;
        for column in 0..MAX_B {
            for (i, out) in self.outputs.iter().enumerate() {
                if out.m == 0 {
                    continue;
                }
                let xp = x.sub_offset(column * self.shape.k, self.shape.k);
                let yp = out.view.sub_offset(column * out.m, out.m);
                let mut blob = KernargBlob::new();
                blob.push_ptr(self.weights[i].buf.as_ptr());
                blob.push_ptr(xp.buf.as_ptr());
                blob.push_ptr(yp.buf.as_ptr());
                blob.push_i32(out.m as i32);
                blob.push_i32(self.shape.k as i32);
                check(blob.len() == 32, "independent scalar anchor ABI")?;
                gpu.launch_kernel_blob(
                    HIP_ANCHOR,
                    [out.m as u32, 1, 1],
                    [32, 1, 1],
                    0,
                    blob.as_mut_slice(),
                )
                .map_err(err)?;
            }
        }
        let reference = self.read(gpu)?;
        // Includes guard/inactive bytes, with full8 reference on itself.
        self.compare(&reference, &reference, MAX_B)?;
        Ok(reference)
    }
}

fn prepared(
    gpu: &Gpu,
    corpus: &JitCorpus,
    fixture: &Fixture,
    x: &GpuTensor,
    n: usize,
    log: &mut Evidence,
) -> PreparedMq6Program {
    // SAFETY: fixture/x owners outlive quiesce and the returned program.
    match unsafe { PreparedMq6Program::prepare(gpu, corpus, &[fixture.invocation(x, n)]) } {
        Ok(value) => value,
        Err(failure) => log.runtime_failure("prepare", failure),
    }
}
fn replay(program: &mut PreparedMq6Program, gpu: &Gpu, log: &mut Evidence) {
    // SAFETY: original allocation owners remain live; runtime performs the
    // producer synchronization and completion join before returning Ok.
    if let Err(failure) = unsafe { program.replay(gpu) } {
        log.runtime_failure("replay", failure)
    }
}
fn finish(program: &mut PreparedMq6Program, log: &mut Evidence) {
    if let Err(failure) = program.quiesce() {
        log.runtime_failure("quiesce", failure)
    }
}

fn identity(report: &Value, nodes: usize) -> Result<Value> {
    let digest = |value: &Value| {
        value
            .as_str()
            .is_some_and(|s| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit()))
    };
    check(digest(&report["ib_sha256"]), "report missing IB identity")?;
    check(
        report
            .get("kernarg_sha256")
            .and_then(Value::as_array)
            .is_some_and(|values| values.len() == nodes && values.iter().all(digest)),
        "report missing kernarg identities",
    )?;
    check(
        report.get("executing_backend").and_then(Value::as_str) == Some("railgun"),
        "backend is not railgun",
    )?;
    check(
        report["bundle_sha256"].as_str() == Some(sha(kernels::QWEN4_LANE_ROWS_PM_GFX1151).as_str())
            && digest(&report["device_elf_sha256"])
            && digest(&report["receipt_sha256"]),
        "report missing/mismatched registered object or certificate identity",
    )?;
    check(
        report["kernel_program"].is_object() && report["plan"].is_object(),
        "report missing typed program/plan",
    )?;
    check(
        report["kernel_program"]["nodes"]
            .as_array()
            .is_some_and(|values| values.len() == nodes),
        "typed program node count mismatch",
    )?;
    Ok(json!({"ib_sha256":report["ib_sha256"],"kernarg_sha256":report["kernarg_sha256"]}))
}

fn check_width(
    gpu: &Gpu,
    corpus: &JitCorpus,
    f: &Fixture,
    n: usize,
    references: [&[Vec<u8>]; 2],
    log: &mut Evidence,
) -> Result<()> {
    let mut program = prepared(gpu, corpus, f, &f.x, n, log);
    let first = program.report();
    let stable = identity(&first, 1)?;
    check(
        first["replay_count"].as_u64() == Some(0),
        "new program replay count",
    )?;
    log.emit(json!({"kind":"prepared","shape":f.shape.name,"batch":n,"report":first}))?;
    for (round, b) in [false, true, false].into_iter().enumerate() {
        f.input(gpu, b)?;
        f.reset(gpu)?;
        replay(&mut program, gpu, log);
        let actual = f.read(gpu)?;
        f.compare(references[usize::from(b)], &actual, n)?;
        let report = program.report();
        check(
            identity(&report, 1)? == stable,
            "retained IB/kernargs changed during same-address ABA",
        )?;
        check(
            report["replay_count"].as_u64() == Some(round as u64 + 1),
            "replay counter mismatch",
        )?;
        log.emit(
            json!({"kind":"exact","shape":f.shape.name,"batch":n,"round":round,
            "input":if b {"B"}else{"A"},"backend":"railgun","passed":true,
            "full_output_sha256":actual.iter().map(|v|sha(v)).collect::<Vec<_>>(),
            "compared_bytes":actual.iter().map(Vec::len).sum::<usize>(),"report":report}),
        )?;
    }
    finish(&mut program, log);
    drop(program);
    Ok(())
}

fn negative_controls(
    gpu: &mut Gpu,
    corpus: &JitCorpus,
    f: &Fixture,
    references: [&[Vec<u8>]; 2],
    log: &mut Evidence,
) -> Result<()> {
    let n = 2;
    f.reset(gpu)?;
    let unwritten = f.read(gpu)?;
    check(
        f.compare(references[0], &unwritten, n).is_err(),
        "unsubmitted/poison negative passed",
    )?;
    let active = f
        .outputs
        .iter()
        .position(|out| out.m > 0)
        .ok_or("no active output")?;
    let mut corruption = f.outputs[active].expected(&references[0][active], n);
    corruption[GUARD * 4..GUARD * 4 + 4].fill(POISON);
    check(
        f.outputs[active]
            .compare(&references[0][active], &corruption, n)
            .is_err(),
        "one-word poison negative passed",
    )?;
    log.emit(
        json!({"kind":"checker_negatives","shape":f.shape.name,"batch":n,
        "no_write_rejected":true,"one_active_word_poison_rejected":true,"passed":true}),
    )?;

    // Fully valid but semantically wrong X binding: must match B and fail A.
    // No invalid address or geometry is ever submitted to the GPU.
    let wrong_x = gpu.upload_f32(&f.b, &[MAX_B * f.shape.k]).map_err(err)?;
    let mut wrong = prepared(gpu, corpus, f, &wrong_x, n, log);
    f.reset(gpu)?;
    replay(&mut wrong, gpu, log);
    let got = f.read(gpu)?;
    f.compare(references[1], &got, n)?;
    check(
        f.compare(references[0], &got, n).is_err(),
        "wrong valid X binding was not detected",
    )?;
    log.emit(
        json!({"kind":"wrong_binding_negative","shape":f.shape.name,"batch":n,
        "expected_A_rejected":true,"actual_B_exact":true,"report":wrong.report(),"passed":true}),
    )?;
    finish(&mut wrong, log);
    drop(wrong);
    drop(wrong_x);

    // Corrupt one actual device output word, then download through the same
    // checker path used by positives. This must fail after an exact result.
    let mut device_corruption = got[active].clone();
    device_corruption[GUARD * 4..GUARD * 4 + 4].fill(POISON);
    gpu.hip
        .memcpy_htod(&f.outputs[active].owner.buf, &device_corruption)
        .map_err(err)?;
    let broken = f.read(gpu)?;
    check(
        f.compare(references[1], &broken, n).is_err(),
        "device output corruption was not detected",
    )?;
    log.emit(
        json!({"kind":"device_corruption_negative","shape":f.shape.name,
        "batch":n,"reference":"B (matched before corruption)",
        "one_active_word_poison_rejected":true,"passed":true}),
    )?;
    Ok(())
}

fn raw_chain_plan(report: &Value) -> Result<()> {
    identity(report, 2)?;
    let segments = report["plan"]["segments"]
        .as_array()
        .ok_or("RAW chain plan has no segments")?;
    check(
        segments.len() == 1
            && segments[0]["first"].as_u64() == Some(0)
            && segments[0]["end"].as_u64() == Some(2),
        "RAW chain did not lower to one two-node segment",
    )?;
    let boundary = &report["plan"]["boundaries"][1];
    check(
        boundary["visibility"]["wait_idle"].as_bool() == Some(true)
            && boundary["visibility"]["acquire"].as_str() == Some("system")
            && boundary["split"].as_bool() == Some(false),
        "RAW consumer boundary is not in-segment WaitIdle + System",
    )?;
    check(
        boundary["edges"].as_array().is_some_and(|edges| {
            edges.iter().any(|edge| {
                edge["producer"].as_u64() == Some(0)
                    && edge["consumer"].as_u64() == Some(1)
                    && edge["hazard"].as_str() == Some("raw")
            })
        }),
        "RAW chain lacks an actual allocation-derived producer0 -> consumer1 edge",
    )
}

/// Two real typed nodes in one retained IB. The second node reads the first
/// node's guarded output view, never a host-computed or copied intermediate.
fn check_raw_chain(gpu: &mut Gpu, corpus: &JitCorpus, log: &mut Evidence) -> Result<()> {
    let first = Fixture::new(
        gpu,
        Shape {
            name: "raw_chain_stage1",
            ms: [256, 0, 0, 0],
            k: 256,
            x4: false,
            native: false,
        },
        0x6d71_0201_2026_1025,
        log,
    )?;
    let second = Fixture::new(
        gpu,
        Shape {
            name: "raw_chain_stage2",
            ms: [19, 0, 0, 0],
            k: 256,
            x4: false,
            native: false,
        },
        0x6d71_0202_2026_1025,
        log,
    )?;
    // second.x is an unused fixture owner; the actual second-node X is the
    // first output view, in both the serial HIP reference and railgun ops.
    let mut middle_refs = Vec::new();
    let mut final_refs = Vec::new();
    for b in [false, true] {
        first.input(gpu, b)?;
        middle_refs.push(first.reference(gpu)?);
        final_refs.push(second.reference_with_x(gpu, &first.outputs[0].view)?);
    }
    for n in [2usize, 8] {
        let ops = [
            first.invocation(&first.x, n),
            second.invocation(&first.outputs[0].view, n),
        ];
        // SAFETY: all original X/W/intermediate/final owners remain alive
        // until quiesce+drop; the allowed cross-node RAW dependency is typed.
        let mut program = match unsafe { PreparedMq6Program::prepare(gpu, corpus, &ops) } {
            Ok(value) => value,
            Err(failure) => log.runtime_failure("raw_chain_prepare", failure),
        };
        let initial = program.report();
        raw_chain_plan(&initial)?;
        let stable = identity(&initial, 2)?;
        check(
            initial["replay_count"].as_u64() == Some(0),
            "RAW chain initial replay count",
        )?;
        log.emit(
            json!({"kind":"raw_chain_prepared","batch":n,"m1":256,"k1":256,
            "m2":19,"k2":256,"report":initial,"second_reads_device_intermediate":true}),
        )?;
        for (round, b) in [false, true, false].into_iter().enumerate() {
            first.input(gpu, b)?;
            first.reset(gpu)?;
            second.reset(gpu)?;
            replay(&mut program, gpu, log);
            let middle = first.read(gpu)?;
            let output = second.read(gpu)?;
            first.compare(&middle_refs[usize::from(b)], &middle, n)?;
            second.compare(&final_refs[usize::from(b)], &output, n)?;
            let report = program.report();
            raw_chain_plan(&report)?;
            check(
                identity(&report, 2)? == stable,
                "RAW chain retained IB/kernargs changed",
            )?;
            check(
                report["replay_count"].as_u64() == Some(round as u64 + 1),
                "RAW chain replay count mismatch",
            )?;
            log.emit(json!({"kind":"raw_chain_exact","batch":n,"round":round,
                "input":if b {"B"}else{"A"},"backend":"railgun","passed":true,
                "middle_full_output_sha256":middle.iter().map(|v|sha(v)).collect::<Vec<_>>(),
                "final_full_output_sha256":output.iter().map(|v|sha(v)).collect::<Vec<_>>(),
                "compared_bytes":middle.iter().chain(&output).map(Vec::len).sum::<usize>(),
                "two_nodes_one_segment":true,"raw_wait_idle_system":true,"report":report}))?;
        }
        finish(&mut program, log);
        drop(program);
    }
    gpu.hip.device_synchronize().map_err(err)?;
    drop(second);
    drop(first);
    Ok(())
}

fn run() -> Result<()> {
    let mut corpus_path = None;
    let mut out = None;
    let mut source_head = None;
    let mut args = std::env::args().skip(1);
    while let Some(key) = args.next() {
        let value = args.next().ok_or_else(|| format!("{key} needs a value"))?;
        match key.as_str() {
            "--corpus" => corpus_path = Some(PathBuf::from(value)),
            "--out" => out = Some(PathBuf::from(value)),
            "--source-head" => source_head = Some(value),
            _ => return Err(format!("unknown argument {key}")),
        }
    }
    let corpus_path = corpus_path.ok_or("--corpus required")?;
    let out = out.ok_or("--out required")?;
    fs::create_dir(&out).map_err(err)?;
    let mut log = Evidence(BufWriter::new(
        File::create(out.join("cases.jsonl")).map_err(err)?,
    ));
    let corpus = JitCorpus::open(&corpus_path).map_err(err)?;
    let mut gpu = Gpu::init().map_err(err)?;
    check(gpu.arch == "gfx1151", "this oracle requires exact gfx1151")?;
    gpu.ensure_capture_stream().map_err(err)?;
    // Original dynamic-K source + raw launch: never follows a new route.
    gpu.ensure_kernel_public(
        "gemv_mq6g256v2_mq6v2",
        kernels::GEMV_MQ6G256V2_SRC,
        HIP_ANCHOR,
    )
    .map_err(err)?;
    let exe = std::env::current_exe().map_err(err)?;
    log.emit(json!({"kind":"provenance","arch":gpu.arch,"corpus":corpus_path,
        "toolchain_pin":corpus.toolchain_pin(),"source_head_claim":source_head,
        "compiled_probe_source_sha256":sha(include_bytes!("mq6_railgun_oracle.rs")),
        "binary_path":exe,"binary_sha256":sha(&fs::read(&exe).map_err(err)?),
        "registered_bundle_sha256":sha(kernels::QWEN4_LANE_ROWS_PM_GFX1151),
        "scope":"explicit prepared/replay for exactly two MQ6 kernels; not capture integration/model/quality/performance",
        "ownership":"caller retains original tensor owners until quiesce+program drop; views do not own allocations"}))?;
    let mut widths = 0usize;
    let mut exact = 0usize;
    for (index, shape) in SHAPES.into_iter().enumerate() {
        let f = Fixture::new(
            &mut gpu,
            shape,
            0x6d71_0100_2026_1025 ^ index as u64,
            &mut log,
        )?;
        f.input(&gpu, false)?;
        let a = f.reference(&gpu)?;
        f.input(&gpu, true)?;
        let b = f.reference(&gpu)?;
        let ns: Vec<usize> = if shape.native {
            vec![2, 8]
        } else {
            (1..=MAX_B).collect()
        };
        for n in ns {
            check_width(&gpu, &corpus, &f, n, [&a, &b], &mut log)?;
            widths += 1;
            exact += 3;
        }
        negative_controls(&mut gpu, &corpus, &f, [&a, &b], &mut log)?;
        gpu.hip.device_synchronize().map_err(err)?;
        // Every prepared object was quiesced/dropped before original owners.
        drop(f);
    }
    check(
        widths == 38 && exact == 114,
        "mandatory case count incomplete",
    )?;
    check_raw_chain(&mut gpu, &corpus, &mut log)?;
    widths += 2;
    exact += 6;
    let summary = json!({"status":"PASS","shape_width_cases":widths,"exact_replays":exact,
        "single_node_cases":38,"raw_chain_cases":2,"raw_chain_replays":6,
        "raw_chain_nodes":2,"raw_chain_segments":1,"raw_wait_idle_system":true,
        "fixtures":SHAPES.len()+2,"mandatory_skipped":0,"wrong_binding_controls":SHAPES.len(),
        "no_write_controls":SHAPES.len(),"one_word_controls":SHAPES.len(),
        "device_corruption_controls":SHAPES.len(),"kernels":[PM_SINGLE,PM_X4],
        "scope":"seeded explicit typed-program arithmetic+retained ABA; no automatic capture, rebinding, model or performance claim"});
    fs::write(
        out.join("summary.json"),
        serde_json::to_vec_pretty(&summary).map_err(err)?,
    )
    .map_err(err)?;
    log.emit(summary)
}

fn main() {
    if let Err(failure) = run() {
        eprintln!("mq6_railgun_oracle: {failure}");
        std::process::exit(1);
    }
}
