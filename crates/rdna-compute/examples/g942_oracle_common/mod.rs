// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Shared plumbing of the exact-gfx942 MoE exact-image byte oracles
//! (`qwen4_g942_moe_wide_oracle`, `qwen4_g942_moe_gemm_oracle`): the PRNG, guarded poison-filled
//! device buffers, raw downloads, the PASS / FAIL and planted-control tallies and the JSON
//! summary. Included with `mod g942_oracle_common;` (not an example itself).

#![allow(dead_code)]

use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use std::collections::BTreeMap;

pub type Res<T> = Result<T, String>;

pub fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

/// Guard words on each side of every output body (must survive every launch).
pub const GUARD: usize = 64;
pub const GUARD_WORD: u32 = 0x7f7f_7f7f;
/// Fill of every output body before a launch: a word an arm fails to write stays this value.
pub const POISON_WORD: u32 = 0xfeed_face;

pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Uniform in [-1, 1).
    pub fn f32(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    pub fn gauss(&mut self) -> f64 {
        let u = self.unit().max(1e-300);
        let v = self.unit();
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
    pub fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len())]
    }
}

pub fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

pub fn i32s_to_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|w| w.to_le_bytes()).collect()
}

pub fn f32s_to_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|w| w.to_le_bytes()).collect()
}

pub fn bytes_to_i32s(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Upload `bytes` as a tensor of `dtype` (a direct allocation, released with [`release`]).
pub fn upload(gpu: &Gpu, bytes: &[u8], dtype: DType) -> Res<GpuTensor> {
    let mut tensor = gpu.upload_raw(bytes, &[bytes.len()]).map_err(err)?;
    tensor.dtype = dtype;
    tensor.shape = vec![bytes.len() / dtype.size()];
    Ok(tensor)
}

/// An F32-typed tensor of the given raw 32-bit words (i32 data travels the same way).
pub fn upload_words(gpu: &Gpu, words: &[u32]) -> Res<GpuTensor> {
    upload(gpu, &words_to_bytes(words), DType::F32)
}

pub fn upload_i32s(gpu: &Gpu, values: &[i32]) -> Res<GpuTensor> {
    upload(gpu, &i32s_to_bytes(values), DType::F32)
}

pub fn upload_f32s(gpu: &Gpu, values: &[f32]) -> Res<GpuTensor> {
    upload(gpu, &f32s_to_bytes(values), DType::F32)
}

pub fn release(gpu: &Gpu, tensor: GpuTensor) {
    let _ = gpu.hip.free(tensor.buf);
}

/// The whole allocation of `tensor`, after the device has drained.
pub fn download(gpu: &Gpu, tensor: &GpuTensor) -> Res<Vec<u8>> {
    gpu.hip.device_synchronize().map_err(err)?;
    let mut bytes = vec![0u8; tensor.buf.size()];
    gpu.hip.memcpy_dtoh(&mut bytes, &tensor.buf).map_err(err)?;
    Ok(bytes)
}

/// A guarded 32-bit-word output: `GUARD` guard words, `words` body words, `GUARD` guard words.
/// `body` is a view of the middle (F32-typed, what the launchers take).
pub struct Out {
    pub full: GpuTensor,
    pub body: GpuTensor,
    pub words: usize,
}

impl Out {
    /// Body filled with [`POISON_WORD`].
    pub fn poisoned(gpu: &Gpu, words: usize) -> Res<Out> {
        Out::with_body(gpu, &vec![POISON_WORD; words])
    }

    /// Body holding `body` (an in-place operand).
    pub fn with_body(gpu: &Gpu, body: &[u32]) -> Res<Out> {
        let mut all = vec![GUARD_WORD; GUARD];
        all.extend_from_slice(body);
        all.extend(std::iter::repeat_n(GUARD_WORD, GUARD));
        let full = upload_words(gpu, &all)?;
        let view = full.sub_offset(GUARD, body.len());
        Ok(Out {
            full,
            body: view,
            words: body.len(),
        })
    }

    /// The body bytes and whether both guards are intact.
    pub fn read(&self, gpu: &Gpu) -> Res<(Vec<u8>, bool)> {
        let bytes = download(gpu, &self.full)?;
        let n = self.words * 4;
        let guard_ok = bytes[..GUARD * 4]
            .chunks_exact(4)
            .chain(bytes[GUARD * 4 + n..].chunks_exact(4))
            .all(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]) == GUARD_WORD);
        Ok((bytes[GUARD * 4..GUARD * 4 + n].to_vec(), guard_ok))
    }

    pub fn release(self, gpu: &Gpu) {
        release(gpu, self.full);
    }
}

/// Number of differing bytes (length difference included).
pub fn diff(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count() + a.len().abs_diff(b.len())
}

#[derive(Default)]
pub struct Tally {
    pub pass: usize,
    pub fail: usize,
    pub detected: usize,
    pub missed: usize,
    pub rows: Vec<Value>,
    /// family -> (pass, fail)
    pub families: BTreeMap<String, (usize, usize)>,
}

impl Tally {
    /// One compared case: every `(name, differing_bytes)` must be 0 and `guards` intact.
    pub fn case(&mut self, family: &str, label: &str, diffs: &[(String, usize)], guards: bool) {
        let ok = guards && diffs.iter().all(|(_, d)| *d == 0);
        let entry = self.families.entry(family.to_owned()).or_default();
        if ok {
            self.pass += 1;
            entry.0 += 1;
        } else {
            self.fail += 1;
            entry.1 += 1;
            let detail: Vec<String> = diffs.iter().map(|(n, d)| format!("{n}={d}")).collect();
            eprintln!(
                "FAIL {family} {label}: {} guards_ok={guards}",
                detail.join(" ")
            );
        }
        if !ok || self.rows.len() < 24 {
            let mut row = json!({"family": family, "case": label, "ok": ok, "guards_ok": guards});
            for (name, d) in diffs {
                row[name.as_str()] = json!(d);
            }
            self.rows.push(row);
        }
    }

    /// A planted control: the module run on a corrupted operand / launch parameter must differ
    /// from the clean reference (any differing body byte, or a broken guard).
    pub fn control(&mut self, name: &str, differing: usize, guards: bool) {
        let detected = differing > 0 || !guards;
        if detected {
            self.detected += 1;
        } else {
            self.missed += 1;
            eprintln!("CONTROL MISSED {name}: no differing byte, guards intact");
        }
        self.rows.push(json!({
            "control": name, "differing_bytes": differing, "guards_ok": guards,
            "detected": detected,
        }));
    }

    pub fn is_pass(&self) -> bool {
        self.fail == 0 && self.missed == 0 && self.pass > 0 && self.detected > 0
    }
}

pub struct Args {
    pub out: Option<String>,
    pub seed: u64,
    pub quick: bool,
}

impl Args {
    pub fn parse(default_seed: u64) -> Args {
        let args: Vec<String> = std::env::args().collect();
        let flag = |name: &str| {
            args.iter()
                .position(|a| a == name)
                .and_then(|i| args.get(i + 1))
                .cloned()
        };
        Args {
            out: flag("--out"),
            seed: flag("--seed")
                .map(|s| s.parse::<u64>().expect("--seed U64"))
                .unwrap_or(default_seed),
            quick: args.iter().any(|a| a == "--quick"),
        }
    }
}

/// `Gpu::init`, or exit 0 with a skip note when the device is not an exact gfx942.
pub fn init_gfx942() -> Gpu {
    let mut gpu = Gpu::init().expect("gpu init");
    if gpu.try_gfx942().is_none() {
        eprintln!("skip: needs an exact gfx942 device");
        std::process::exit(0);
    }
    gpu
}

/// Print the result line, write `--out`, return the process exit code.
pub fn finish(oracle: &str, args: &Args, tally: &Tally, image_var: &str, extra: Value) -> i32 {
    let image = std::env::var(image_var).ok().filter(|p| !p.is_empty());
    println!(
        "RESULT cases PASS {} FAIL {} | controls DETECTED {} MISSED {}",
        tally.pass, tally.fail, tally.detected, tally.missed
    );
    for (family, (pass, fail)) in &tally.families {
        println!("  {family}: PASS {pass} FAIL {fail}");
    }
    if let Some(path) = &args.out {
        let report = json!({
            "oracle": oracle, "seed": args.seed, "quick": args.quick,
            "image_env": image_var, "image": image,
            "cases_pass": tally.pass, "cases_fail": tally.fail,
            "controls_detected": tally.detected, "controls_missed": tally.missed,
            "families": tally.families.iter()
                .map(|(k, (p, f))| (k.clone(), json!({"pass": p, "fail": f})))
                .collect::<serde_json::Map<String, Value>>(),
            "extra": extra,
            "rows": tally.rows,
        });
        std::fs::write(path, serde_json::to_string_pretty(&report).unwrap()).expect("write --out");
    }
    let ok = tally.is_pass();
    println!("{}", if ok { "PASS" } else { "FAIL" });
    if ok {
        0
    } else {
        1
    }
}

/// Describe the module arm for the banner.
pub fn module_banner(module: &str, image_var: &str) -> String {
    match std::env::var(image_var).ok().filter(|p| !p.is_empty()) {
        Some(path) => format!("module arm: code object {path} ({image_var})"),
        None => format!("module arm: hipcc source (module {module})"),
    }
}
