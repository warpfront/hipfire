// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#![allow(dead_code)]
//! B4-P1 (SOL-B4-B5 §2.5) arithmetic / address proof for the gfx1100 symmetric IU4 MoE kernels.
//!
//! Pure CPU enumeration over the *frozen contract models* of the kernel (no device, no G1):
//!
//! * W3: every 24-bit chunk with random high-byte junk against the public
//!   `qwen4_moe_sym::w3::spread_model` and an independent per-code reference `(2c-7)&15`.
//! * W2: every 16-bit chunk with random high-half junk against an independent bit-parallel
//!   spread network and the per-code reference `(2c-3)&15`.
//! * MQ4: all 16 stored nibbles at all 8 positions against `x ^ REBIAS` (`iu4_fold::REBIAS`);
//!   only this codec rebiases.
//! * Signed dot bounds derived from the observed nibble ranges (x is signed 4-bit), the magic
//!   conversion window for every integer in the bound, and the epoch fold order witness
//!   (ascending vs reversed differ in F32 bits, so the ReversedEpoch negative is distinguishable).
//! * Per-lane / per-epoch byte-address trace of every weight, scale and activation load against the
//!   exact tensor extent (FN sizes, odd K640 Down, a small grid), planar W3 last-dword overread,
//!   16-bit scale tail, native gfx11 C-row mapping, `ds_swizzle` BROADCAST16 selector `2j`,
//!   and the 64-bit expert pointer add.
//!
//! This is a model-level proof; it does not execute the emitted object. The emitted object is
//! audited separately (disassembly / M7 / descriptor) by `emit_qwen4_moe_sym_gfx1100`.
//!
//! Standalone: `cargo run -p peacemaker-isa --release --features lab --example qwen4_moe_sym_gfx1100_p1 -- [--seed N] [--receipt FILE]`
use peacemaker_isa::kernels::{iu4_fold, qwen4_moe_sym::w3};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

/// Result of [`prove`]: the machine-readable receipt and every failed obligation.
pub struct Outcome {
    pub receipt: Value,
    pub failures: Vec<String>,
}

/// The gfx1100 kernel module's public CPU mirrors and address helpers (codec "mq4"|"w2"|"w3",
/// kind "gate_up"|"down"), passed by the emitter so P1 enumerates the kernel's own models.
pub struct Kernel<'a> {
    pub w2_spread: &'a dyn Fn(u32) -> u32,
    pub w3_spread: &'a dyn Fn(u32) -> u32,
    pub mq4_unpack: &'a dyn Fn(u32) -> u32,
    pub code_offset: &'a dyn Fn(&str, &str, u32) -> u32,
    pub scale_offset: &'a dyn Fn(&str, &str, u32) -> u32,
    pub code_loads: &'a dyn Fn(&str) -> Vec<(u32, u8)>,
    pub row_code_bytes: &'a dyn Fn(&str, u32) -> u32,
    pub row_scale_bytes: &'a dyn Fn(&str, u32) -> u32,
    pub tensor_bytes: &'a dyn Fn(&str, u32, u32) -> u64,
    pub dot_bound: &'a dyn Fn(&str) -> u32,
    pub scale_row: &'a dyn Fn(u32, u32, bool) -> u32,
    pub accumulator_row: &'a dyn Fn(u32, u32) -> u32,
    pub swizzle_lane: &'a dyn Fn(u32) -> u32,
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn signed_nibble(n: u32) -> i32 {
    let n = (n & 15) as i32;
    if n >= 8 {
        n - 16
    } else {
        n
    }
}

// ------------------------------------------------------------------------------------------
// Unpack models

/// Independent per-code reference: code `k` of `x` is `bits` wide; nibble `k` is `(2c - bias) & 15`.
fn reference(x: u32, bits: u32, bias: u32) -> u32 {
    let mask = (1u32 << bits) - 1;
    (0..8u32).fold(0u32, |acc, k| {
        acc | (((2 * ((x >> (bits * k)) & mask)).wrapping_sub(bias) & 15) << (4 * k))
    })
}

/// W3 reference `(2c - 7) & 15`.
fn w3_reference(x: u32) -> u32 {
    reference(x, 3, 7)
}

/// W2 reference `(2c - 3) & 15` over the low 16 bits.
fn w2_reference(x: u32) -> u32 {
    reference(x & 0xffff, 2, 3)
}

/// Bit-parallel W2 spread of one u16 chunk (high half ignored): each 2-bit code to the low half
/// of its own nibble, then `(2c-3)&15` by `((2c | 8) - 3) ^ 8` per nibble (no inter-nibble borrow).
fn w2_model(chunk: u32) -> u32 {
    let x = chunk & 0xffff;
    let x = (x | (x << 8)) & 0x00ff_00ff;
    let x = (x | (x << 4)) & 0x0f0f_0f0f;
    let x = (x | (x << 2)) & 0x3333_3333;
    let s = x << 1;
    ((s | 0x8888_8888).wrapping_sub(0x3333_3333)) ^ 0x8888_8888
}

/// MQ4: eight stored nibbles, unsigned minus eight, is the XOR rebias.
fn mq4_model(dword: u32) -> u32 {
    dword ^ iu4_fold::REBIAS
}

struct Sweep {
    chunks: u64,
    mismatches: u64,
    junk_dependent: u64,
    first_mismatch: Option<(u32, u32, u32)>,
    signed_min: i32,
    signed_max: i32,
    seen_nibbles: u16,
    output_sha256: String,
}

fn sweep(
    bits: u32,
    junk_mask: u32,
    rng: &mut Rng,
    model: &dyn Fn(u32) -> u32,
    want: &dyn Fn(u32) -> u32,
) -> Sweep {
    let mut s = Sweep {
        chunks: 0,
        mismatches: 0,
        junk_dependent: 0,
        first_mismatch: None,
        signed_min: i32::MAX,
        signed_max: i32::MIN,
        seen_nibbles: 0,
        output_sha256: String::new(),
    };
    let mut hash = Sha256::new();
    let mut buf: Vec<u8> = Vec::with_capacity(1 << 16);
    for x in 0..(1u64 << bits) {
        let x = x as u32;
        let junk = (rng.next() as u32) & junk_mask;
        let got = model(x | junk);
        let clean = model(x);
        let expected = want(x);
        s.chunks += 1;
        if got != clean {
            s.junk_dependent += 1;
        }
        if got != expected || clean != expected {
            s.mismatches += 1;
            s.first_mismatch.get_or_insert((x | junk, got, expected));
        }
        for k in 0..8 {
            let n = (got >> (4 * k)) & 15;
            s.seen_nibbles |= 1 << n;
            let v = signed_nibble(n);
            s.signed_min = s.signed_min.min(v);
            s.signed_max = s.signed_max.max(v);
        }
        buf.extend_from_slice(&got.to_le_bytes());
        if buf.len() >= 1 << 16 {
            hash.update(&buf);
            buf.clear();
        }
    }
    hash.update(&buf);
    s.output_sha256 = hex(&hash.finalize());
    s
}

fn sweep_json(name: &str, s: &Sweep, expected_bound: i32, failures: &mut Vec<String>) -> Value {
    let max_w = s.signed_min.abs().max(s.signed_max.abs());
    // X is a signed 4-bit activation: |x| <= 8. One K128 epoch sums 128 products.
    let bound = 128 * max_w * 8;
    if s.mismatches != 0 {
        failures.push(format!(
            "P1 {name}: {} unpack mismatches, first {:?}",
            s.mismatches, s.first_mismatch
        ));
    }
    if s.junk_dependent != 0 {
        failures.push(format!(
            "P1 {name}: {} chunks depend on the junk above the chunk",
            s.junk_dependent
        ));
    }
    if bound != expected_bound {
        failures.push(format!(
            "P1 {name}: derived dot bound {bound}, frozen {expected_bound}"
        ));
    }
    json!({
        "chunks_enumerated": s.chunks, "mismatches": s.mismatches, "junk_dependent_chunks": s.junk_dependent,
        "first_mismatch": s.first_mismatch.map(|(x, g, w)| json!({"input": format!("{x:#010x}"), "got": format!("{g:#010x}"), "want": format!("{w:#010x}")})),
        "signed_weight_min": s.signed_min, "signed_weight_max": s.signed_max,
        "nibble_values_seen_mask": format!("{:#06x}", s.seen_nibbles),
        "dot_bound_derived": bound, "dot_bound_frozen": expected_bound,
        "output_sha256": s.output_sha256,
    })
}

fn prove_mq4(rng: &mut Rng, model: &dyn Fn(u32) -> u32, failures: &mut Vec<String>) -> Value {
    let mut checks = 0u64;
    let mut bad = 0u64;
    let mut min = i32::MAX;
    let mut max = i32::MIN;
    let mut words: Vec<u32> = (0..16u32).map(|n| n * 0x1111_1111).collect();
    for _ in 0..3 {
        let background = rng.next() as u32;
        for p in 0..8u32 {
            for n in 0..16u32 {
                words.push((background & !(0xf << (4 * p))) | (n << (4 * p)));
            }
        }
    }
    for dw in words {
        let got = model(dw);
        for q in 0..8u32 {
            let stored = ((dw >> (4 * q)) & 15) as i32;
            let v = signed_nibble(got >> (4 * q));
            checks += 1;
            min = min.min(v);
            max = max.max(v);
            if v != stored - 8 {
                bad += 1;
            }
        }
    }
    // The nibble values observed at one fixed position are exactly the 16 stored values.
    let mut per_value = [false; 16];
    for n in 0..16u32 {
        per_value[n as usize] = signed_nibble(model(n) & 15) == n as i32 - 8;
    }
    let all16 = per_value.iter().all(|v| *v);
    let bound = 128 * min.abs().max(max.abs()) * 8;
    if bad != 0 || !all16 {
        failures.push(format!("P1 mq4: {bad} nibble mismatches, all16={all16}"));
    }
    if bound != 8192 {
        failures.push(format!("P1 mq4: derived dot bound {bound}, frozen 8192"));
    }
    if iu4_fold::REBIAS != 0x8888_8888 {
        failures.push(format!(
            "P1 mq4: iu4_fold::REBIAS = {:#x}",
            iu4_fold::REBIAS
        ));
    }
    json!({
        "stored_nibble_values_checked": 16, "all_16_nibbles_exact": all16,
        "nibble_position_checks": checks, "mismatches": bad,
        "signed_weight_min": min, "signed_weight_max": max,
        "dot_bound_derived": bound, "dot_bound_frozen": 8192,
        "rebias": format!("{:#010x}", iu4_fold::REBIAS),
        "rebias_applies_to": "MQ4 only (W2/W3 outputs are already signed nibbles)",
    })
}

fn prove_magic(failures: &mut Vec<String>) -> Value {
    let base = f32::from_bits(iu4_fold::MAGIC);
    if base != 12_582_912.0 {
        failures.push(format!(
            "P1 magic: MAGIC {:#x} is {base}, not 12582912.0",
            iu4_fold::MAGIC
        ));
    }
    let bound = 8192i32;
    let mut checked = 0u64;
    for d in -bound..=bound {
        let v = f32::from_bits(iu4_fold::MAGIC.wrapping_add(d as u32)) - base;
        checked += 1;
        if v != d as f32 {
            failures.push(format!(
                "P1 magic: integer {d} does not round trip through the magic window ({v})"
            ));
            break;
        }
    }
    json!({
        "magic": format!("{:#010x}", iu4_fold::MAGIC), "integers_round_tripped": checked,
        "window": "|D| < 2^22", "largest_bound": bound,
    })
}

// ------------------------------------------------------------------------------------------
// Fold order

fn fold(order: &[usize], scale: &[f32], dot: &[i32]) -> f32 {
    let mut sum = 0.0f32;
    for &h in order {
        sum = scale[h].mul_add(dot[h] as f32, sum);
    }
    sum
}

fn random_scale(rng: &mut Rng) -> f32 {
    let exp = (rng.next() % 9) as i32 - 6;
    f32::from_bits((((127 + exp) as u32) << 23) | (rng.next() as u32 & 0x007f_ffff))
}

fn fold_witness(epochs: usize, bound: i32, rng: &mut Rng) -> Option<Value> {
    for trial in 0..4_000_000u64 {
        let scale: Vec<f32> = (0..epochs)
            .map(|_| random_scale(rng) * random_scale(rng))
            .collect();
        let dot: Vec<i32> = (0..epochs)
            .map(|_| (rng.next() % (2 * bound as u64 + 1)) as i32 - bound)
            .collect();
        let up: Vec<usize> = (0..epochs).collect();
        let down: Vec<usize> = (0..epochs).rev().collect();
        let (a, d) = (fold(&up, &scale, &dot), fold(&down, &scale, &dot));
        if a.to_bits() != d.to_bits() {
            return Some(json!({
                "epochs": epochs, "trial": trial,
                "scale_rn_dact_dw_bits": scale.iter().map(|s| format!("{:#010x}", s.to_bits())).collect::<Vec<_>>(),
                "dot": dot, "ascending_bits": format!("{:#010x}", a.to_bits()), "reversed_bits": format!("{:#010x}", d.to_bits()),
            }));
        }
    }
    None
}

fn prove_fold(rng: &mut Rng, failures: &mut Vec<String>) -> Value {
    let mut witnesses = Vec::new();
    for epochs in [5usize, 20] {
        match fold_witness(epochs, 8192, rng) {
            Some(w) => witnesses.push(w),
            None => failures.push(format!(
                "P1 fold: no ascending/reversed F32 difference found for {epochs} epochs"
            )),
        }
    }
    json!({
        "rule": "S[h+1] = fma(RN(d_act[h]*d_w[h]), float(D[h]), S[h]) from +0, ascending epoch, no tree, no contraction across the scale multiply",
        "reversed_epoch_negative_distinguishable": witnesses.len() == 2,
        "witnesses": witnesses,
    })
}

// ------------------------------------------------------------------------------------------
// Address / epoch / scale trace

#[derive(Clone, Copy, PartialEq, Debug)]
enum Codec {
    Mq4,
    W2,
    W3,
}
#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind {
    GateUp,
    Down,
}

#[derive(Clone, Copy, Debug)]
struct Cfg {
    codec: Codec,
    kind: Kind,
    m: u32,
    k: u32,
}

impl Cfg {
    fn name(self) -> String {
        format!("{:?}_{:?}_m{}_k{}", self.codec, self.kind, self.m, self.k).to_lowercase()
    }
    fn codec_name(self) -> &'static str {
        match self.codec {
            Codec::Mq4 => "mq4",
            Codec::W2 => "w2",
            Codec::W3 => "w3",
        }
    }
    fn kind_name(self) -> &'static str {
        match self.kind {
            Kind::GateUp => "gate_up",
            Kind::Down => "down",
        }
    }
    fn epochs(self) -> u64 {
        u64::from(self.k / 128)
    }
    fn tensors(self) -> u32 {
        if self.kind == Kind::GateUp {
            2
        } else {
            1
        }
    }
    fn rows_per_tensor(self) -> u32 {
        if self.kind == Kind::GateUp {
            self.m / 2
        } else {
            self.m
        }
    }
    /// Frozen admission: positive K multiples, output-row divisibility. No generic tail.
    fn admit(self) -> Result<(), String> {
        let k_mult = if self.kind == Kind::GateUp { 256 } else { 128 };
        if self.k == 0 || self.k % k_mult != 0 {
            return Err(format!(
                "{}: K {} is not a positive multiple of {k_mult}",
                self.name(),
                self.k
            ));
        }
        if self.m == 0 || self.m % 64 != 0 {
            return Err(format!(
                "{}: M {} is not a positive multiple of 64",
                self.name(),
                self.m
            ));
        }
        Ok(())
    }
    fn code_row_bytes(self) -> u64 {
        let e = self.epochs();
        match (self.codec, self.kind) {
            (Codec::W3, _) => 48 * e,
            (Codec::W2, _) => 32 * e,
            (Codec::Mq4, Kind::GateUp) => 136 * (e / 2),
            (Codec::Mq4, Kind::Down) => 68 * e,
        }
    }
    fn code_plane_bytes(self) -> u64 {
        u64::from(self.m) * self.code_row_bytes()
    }
    fn scale_row_bytes(self) -> u64 {
        2 * self.epochs()
    }
    /// Whole expert tensor: the buffer extent every load must stay inside.
    fn extent(self) -> u64 {
        match self.codec {
            Codec::Mq4 => self.code_plane_bytes(),
            _ => self.code_plane_bytes() + u64::from(self.m) * self.scale_row_bytes(),
        }
    }
    /// `(offset, load length, useful length)` of each code load of one row epoch.
    fn code_loads(self, row: u64, e: u64) -> Vec<(u64, u64, u64)> {
        let rb = row * self.code_row_bytes();
        match (self.codec, self.kind) {
            (Codec::W3, _) => (0..8u64)
                .flat_map(|i| [6 * i, 6 * i + 3])
                .map(|o| (rb + 48 * e + o, 4, 3))
                .collect(),
            (Codec::W2, _) => (0..8u64)
                .flat_map(|i| [4 * i, 4 * i + 2])
                .map(|o| (rb + 32 * e + o, 4, 2))
                .collect(),
            (Codec::Mq4, Kind::Down) => (0..4u64)
                .map(|q| (rb + 68 * e + 4 + 16 * q, 16, 16))
                .collect(),
            (Codec::Mq4, Kind::GateUp) => (0..4u64)
                .map(|q| (rb + (e >> 1) * 136 + 8 + (e & 1) * 64 + 16 * q, 16, 16))
                .collect(),
        }
    }
    /// First byte of the epoch's code span.
    fn epoch_code_start(self, row: u64, e: u64) -> u64 {
        let rb = row * self.code_row_bytes();
        match (self.codec, self.kind) {
            (Codec::W3, _) => rb + 48 * e,
            (Codec::W2, _) => rb + 32 * e,
            (Codec::Mq4, Kind::Down) => rb + 68 * e + 4,
            (Codec::Mq4, Kind::GateUp) => rb + (e >> 1) * 136 + 8 + (e & 1) * 64,
        }
    }
    fn epoch_code_bytes(self) -> u64 {
        match self.codec {
            Codec::W3 => 48,
            Codec::W2 => 32,
            Codec::Mq4 => 64,
        }
    }
    /// 16-bit scale load of one row epoch (always 2 bytes).
    fn scale_offset(self, row: u64, e: u64) -> u64 {
        match (self.codec, self.kind) {
            (Codec::Mq4, Kind::Down) => row * self.code_row_bytes() + 68 * e,
            (Codec::Mq4, Kind::GateUp) => {
                row * self.code_row_bytes() + (e >> 1) * 136 + (e & 1) * 4
            }
            _ => self.code_plane_bytes() + row * self.scale_row_bytes() + 2 * e,
        }
    }
}

struct Trace {
    code_loads: u64,
    scale_loads: u64,
    code_max_end: u64,
    scale_max_end: u64,
    overread_loads: u64,
    pair_overrun: u64,
    x_loads: u64,
    trace_sha256: String,
    reversed_trace_sha256: String,
}

fn check_config(cfg: Cfg, x_src_rows: &[u32], kernel: Option<&Kernel>) -> Result<Trace, String> {
    cfg.admit()?;
    if let Some(k) = kernel {
        let (cn, kn) = (cfg.codec_name(), cfg.kind_name());
        let scale_row = if cfg.codec == Codec::Mq4 {
            0
        } else {
            cfg.scale_row_bytes()
        };
        if u64::from((k.row_code_bytes)(cn, cfg.k)) != cfg.code_row_bytes() {
            return Err(format!(
                "{}: kernel row_code_bytes differs from the model",
                cfg.name()
            ));
        }
        if u64::from((k.row_scale_bytes)(cn, cfg.k)) != scale_row {
            return Err(format!(
                "{}: kernel row_scale_bytes differs from the model",
                cfg.name()
            ));
        }
        if (k.tensor_bytes)(cn, cfg.m, cfg.k) != cfg.extent() {
            return Err(format!(
                "{}: kernel tensor_bytes differs from the model extent",
                cfg.name()
            ));
        }
        for e in 0..cfg.epochs() {
            let start = cfg.epoch_code_start(0, e);
            if u64::from((k.code_offset)(cn, kn, e as u32)) != start {
                return Err(format!(
                    "{}: kernel code_offset(epoch {e}) differs from the model",
                    cfg.name()
                ));
            }
            let so = cfg.scale_offset(0, e);
            let rel = if cfg.codec == Codec::Mq4 {
                so
            } else {
                so - cfg.code_plane_bytes()
            };
            if u64::from((k.scale_offset)(cn, kn, e as u32)) != rel {
                return Err(format!(
                    "{}: kernel scale_offset(epoch {e}) differs from the model",
                    cfg.name()
                ));
            }
            let mine: Vec<(u64, u64)> = cfg
                .code_loads(0, e)
                .iter()
                .map(|&(o, len, _)| (o - start, len))
                .collect();
            let theirs: Vec<(u64, u64)> = (k.code_loads)(cn)
                .iter()
                .map(|&(o, w)| (u64::from(o), 4 * u64::from(w)))
                .collect();
            if mine != theirs {
                return Err(format!(
                    "{}: kernel code_loads differ from the model ({theirs:?} vs {mine:?})",
                    cfg.name()
                ));
            }
        }
    }
    let epochs = cfg.epochs();
    let extent = cfg.extent();
    let name = cfg.name();
    if extent > u64::from(u32::MAX) {
        return Err(format!(
            "{name}: tensor extent {extent} exceeds a 32-bit buffer range"
        ));
    }
    let mut t = Trace {
        code_loads: 0,
        scale_loads: 0,
        code_max_end: 0,
        scale_max_end: 0,
        overread_loads: 0,
        pair_overrun: 0,
        x_loads: 0,
        trace_sha256: String::new(),
        reversed_trace_sha256: String::new(),
    };
    let half = u64::from(cfg.m / 2);
    for tensor in 0..cfg.tensors() {
        let trow = if cfg.kind == Kind::GateUp {
            u64::from(tensor) * half
        } else {
            0
        };
        for rb in (0..cfg.rows_per_tensor()).step_by(16) {
            for lr in 0..16u32 {
                let wrow = u64::from(rb + lr) + trow;
                // The scale rows of both halves; the weight row is the same in both halves.
                for hi in 0..2u32 {
                    let srow = u64::from(rb + ((lr & 14) | hi)) + trow;
                    for e in 0..epochs {
                        if hi == 0 {
                            let loads = cfg.code_loads(wrow, e);
                            // Useful spans tile the epoch's code bytes exactly once.
                            let mut spans: Vec<(u64, u64)> =
                                loads.iter().map(|&(o, _, u)| (o, u)).collect();
                            spans.sort_unstable();
                            let start = cfg.epoch_code_start(wrow, e);
                            let mut cursor = start;
                            for (o, u) in spans {
                                if o != cursor {
                                    return Err(format!("{name}: row {wrow} epoch {e}: useful code bytes not contiguous at {o}, expected {cursor}"));
                                }
                                cursor += u;
                            }
                            if cursor != start + cfg.epoch_code_bytes() {
                                return Err(format!(
                                    "{name}: row {wrow} epoch {e}: code coverage {} != {}",
                                    cursor - start,
                                    cfg.epoch_code_bytes()
                                ));
                            }
                            let row_end = (wrow + 1) * cfg.code_row_bytes();
                            for (o, len, _) in loads {
                                let end = o + len;
                                if end > extent {
                                    return Err(format!(
                                        "{name}: code load [{o},{end}) beyond extent {extent}"
                                    ));
                                }
                                if cfg.codec != Codec::Mq4 && end > row_end {
                                    // Only the W3 last-chunk dword of the last epoch reads one byte of the next
                                    // row's codes (or, for the last row, the first byte of this tensor's scale plane).
                                    let over = end - row_end;
                                    let allowed = if cfg.codec == Codec::W3 { 1 } else { 2 };
                                    if over != allowed || e != epochs - 1 {
                                        return Err(format!("{name}: code load [{o},{end}) overreads its row by {over} bytes at epoch {e}"));
                                    }
                                    if wrow == u64::from(cfg.m) - 1 && row_end + over > extent {
                                        return Err(format!(
                                            "{name}: last-row overread leaves the tensor"
                                        ));
                                    }
                                    t.overread_loads += 1;
                                } else if cfg.codec == Codec::Mq4 && end > row_end {
                                    return Err(format!("{name}: MQ4 code load [{o},{end}) crosses its row end {row_end}"));
                                }
                                t.code_max_end = t.code_max_end.max(end);
                                t.code_loads += 1;
                            }
                        }
                        let so = cfg.scale_offset(srow, e);
                        let se = so + 2;
                        if se > extent {
                            return Err(format!(
                                "{name}: scale load [{so},{se}) beyond extent {extent}"
                            ));
                        }
                        if cfg.codec != Codec::Mq4 {
                            let lo = cfg.code_plane_bytes() + srow * cfg.scale_row_bytes();
                            if so < lo || se > lo + cfg.scale_row_bytes() {
                                return Err(format!(
                                    "{name}: scale load [{so},{se}) outside scale row {srow}"
                                ));
                            }
                            // A dword scale-pair load (even epochs) would read past the tensor on an odd tail.
                            if e % 2 == 0 {
                                t.pair_overrun = t.pair_overrun.max((so + 4).saturating_sub(extent))
                            }
                        }
                        t.scale_max_end = t.scale_max_end.max(se);
                        t.scale_loads += 1;
                    }
                }
            }
        }
    }
    if cfg.codec != Codec::Mq4 {
        if t.scale_max_end != extent {
            return Err(format!(
                "{name}: final scale ends at {}, extent {extent}",
                t.scale_max_end
            ));
        }
        let want = 2 * (epochs % 2);
        if t.pair_overrun != want {
            return Err(format!(
                "{name}: dword scale-pair overrun {} != {want} for {epochs} epochs",
                t.pair_overrun
            ));
        }
    } else if t.code_max_end != extent {
        return Err(format!(
            "{name}: MQ4 loads end at {}, extent {extent}",
            t.code_max_end
        ));
    }
    // Activation blocks: [E][x_src_rows][72]; d f32 @0, 64 data bytes @8.
    for &xs in x_src_rows {
        let x_extent = epochs * u64::from(xs) * 72;
        let mut max_end = 0;
        for r in [0u64, u64::from(xs) - 1] {
            for e in 0..epochs {
                let b = (e * u64::from(xs) + r) * 72;
                for (o, len) in [(b, 4u64), (b + 8, 64)] {
                    if o + len > x_extent {
                        return Err(format!(
                            "{name}: X load [{o},{}) beyond {x_extent} (x_src_rows {xs})",
                            o + len
                        ));
                    }
                    max_end = max_end.max(o + len);
                    t.x_loads += 1;
                }
            }
        }
        if max_end != x_extent {
            return Err(format!(
                "{name}: X loads end at {max_end}, extent {x_extent} (x_src_rows {xs})"
            ));
        }
    }
    // Epoch trace for row 0 and the last row, ascending and reversed (operand/scale pairing kept).
    let last = u64::from(cfg.m) - 1;
    let mut asc = Sha256::new();
    let mut desc = Sha256::new();
    let mut up = Vec::new();
    let mut down = Vec::new();
    for e in 0..epochs {
        let rec = [
            e,
            cfg.epoch_code_start(0, e),
            cfg.scale_offset(0, e),
            cfg.epoch_code_start(last, e),
            cfg.scale_offset(last, e),
        ];
        up.push(rec);
    }
    for e in (0..epochs).rev() {
        down.push([
            e,
            cfg.epoch_code_start(0, e),
            cfg.scale_offset(0, e),
            cfg.epoch_code_start(last, e),
            cfg.scale_offset(last, e),
        ]);
    }
    for r in &up {
        for v in r {
            asc.update(v.to_le_bytes())
        }
    }
    for r in &down {
        for v in r {
            desc.update(v.to_le_bytes())
        }
    }
    let mut a = up.clone();
    let mut d = down.clone();
    a.sort_unstable();
    d.sort_unstable();
    if a != d {
        return Err(format!(
            "{name}: reversed epoch trace changes an operand/scale pairing"
        ));
    }
    if epochs > 1 && up == down {
        return Err(format!("{name}: reversed trace equals ascending trace"));
    }
    t.trace_sha256 = hex(&asc.finalize());
    t.reversed_trace_sha256 = hex(&desc.finalize());
    Ok(t)
}

fn prove_addresses(failures: &mut Vec<String>, kernel: Option<&Kernel>) -> Value {
    let xs: &[u32] = &[1, 16, 17, 1536, 4096];
    let mut configs = Vec::new();
    for codec in [Codec::Mq4, Codec::W2, Codec::W3] {
        // FN sizes: GateUp M=2*640, K=2560; Down M=2560, K=640 (odd epoch count).
        let mut list = vec![
            Cfg {
                codec,
                kind: Kind::GateUp,
                m: 1280,
                k: 2560,
            },
            Cfg {
                codec,
                kind: Kind::Down,
                m: 2560,
                k: 640,
            },
            Cfg {
                codec,
                kind: Kind::Down,
                m: 2560,
                k: 2560,
            },
        ];
        for m in [64u32, 128] {
            for k in (128..=1536u32).step_by(128) {
                list.push(Cfg {
                    codec,
                    kind: Kind::Down,
                    m,
                    k,
                });
                if k % 256 == 0 {
                    list.push(Cfg {
                        codec,
                        kind: Kind::GateUp,
                        m,
                        k,
                    })
                }
            }
        }
        for cfg in list {
            let fn_size = matches!(
                (cfg.kind, cfg.m, cfg.k),
                (Kind::GateUp, 1280, 2560) | (Kind::Down, 2560, 640)
            );
            match check_config(cfg, xs, kernel) {
                Ok(t) => configs.push(json!({
                    "config": cfg.name(), "fn_size": fn_size, "epochs": cfg.epochs(), "extent_bytes": cfg.extent(),
                    "code_loads": t.code_loads, "scale_loads": t.scale_loads, "x_loads": t.x_loads,
                    "code_max_end": t.code_max_end, "scale_max_end": t.scale_max_end, "w3_last_dword_overread_loads": t.overread_loads,
                    "dword_scale_pair_overrun_bytes": t.pair_overrun,
                    "scale_load_bits": 16, "epoch_trace_sha256": t.trace_sha256, "reversed_epoch_trace_sha256": t.reversed_trace_sha256,
                })),
                Err(e) => failures.push(format!("P1 address {e}")),
            }
        }
    }
    // Admission refusals: no generic tail.
    let mut refusals = 0;
    for bad in [
        Cfg {
            codec: Codec::W3,
            kind: Kind::GateUp,
            m: 2560,
            k: 640,
        },
        Cfg {
            codec: Codec::W3,
            kind: Kind::Down,
            m: 2560,
            k: 0,
        },
        Cfg {
            codec: Codec::W2,
            kind: Kind::Down,
            m: 2560,
            k: 700,
        },
        Cfg {
            codec: Codec::Mq4,
            kind: Kind::Down,
            m: 100,
            k: 640,
        },
    ] {
        if bad.admit().is_ok() {
            failures.push(format!("P1 admission accepted {}", bad.name()))
        } else {
            refusals += 1
        }
    }
    let total_code: u64 = configs
        .iter()
        .map(|c| c["code_loads"].as_u64().unwrap_or(0))
        .sum();
    let total_scale: u64 = configs
        .iter()
        .map(|c| c["scale_loads"].as_u64().unwrap_or(0))
        .sum();
    json!({
        "assumed_layout": {
            "w3": "codes plane [M][48*E] then f16 scale plane [M][E]; chunk dwords at 6i and 6i+3 (low 24 bits used)",
            "w2": "codes plane [M][32*E] then f16 scale plane [M][E]; u16 chunks at 4i and 4i+2",
            "mq4_down": "[M][E*68]: 4-byte header (scale f16 low) then 64 code bytes per epoch",
            "mq4_gate_up": "[M][E/2*136]: per 2 epochs two 4-byte headers (scale f16 low) then 2x64 code bytes",
            "gate_up_rows": "one expert tensor of M rows: gate rows [0,M/2), up rows [M/2,M)",
            "x": "[E][x_src_rows][72]: d f32 @0, 64 data bytes @8",
        },
        "configs_checked": configs.len(), "code_load_checks": total_code, "scale_load_checks": total_scale,
        "admission_refusals": refusals, "configs": configs,
    })
}

// ------------------------------------------------------------------------------------------
// gfx11 native C layout, scale selector, 64-bit expert pointer

fn prove_layout(failures: &mut Vec<String>, kernel: Option<&Kernel>) -> Value {
    let local_sr = |lr: u32, hi: u32, wrong: bool| {
        if wrong {
            8 * hi + (lr & 7)
        } else {
            (lr & 14) | hi
        }
    };
    let local_ar = |j: u32, hi: u32| 2 * j + hi;
    let local_sw = |j: u32| 2 * j;
    let sr: &dyn Fn(u32, u32, bool) -> u32 = match kernel {
        Some(k) => k.scale_row,
        None => &local_sr,
    };
    let ar: &dyn Fn(u32, u32) -> u32 = match kernel {
        Some(k) => k.accumulator_row,
        None => &local_ar,
    };
    let sw: &dyn Fn(u32) -> u32 = match kernel {
        Some(k) => k.swizzle_lane,
        None => &local_sw,
    };
    // The formulas themselves (the kernel's helpers are held to the frozen gfx11 mapping).
    let mut formula_errors = 0u32;
    for hi in 0..2u32 {
        for lr in 0..16u32 {
            if sr(lr, hi, false) != ((lr & 14) | hi) || sr(lr, hi, true) != 8 * hi + (lr & 7) {
                formula_errors += 1
            }
        }
        for j in 0..8u32 {
            if ar(j, hi) != 2 * j + hi || sw(j) != 2 * j {
                formula_errors += 1
            }
        }
    }
    if formula_errors != 0 {
        failures.push(format!(
            "P1 layout: {formula_errors} helper formulas differ from the frozen gfx11 mapping"
        ))
    }
    // WMMA 16x16x16 C: element (row r, column n) lives in lane n + 16*(r&1), accumulator r>>1.
    let mut seen = std::collections::BTreeSet::new();
    let mut layout_bad = 0u32;
    for r in 0..16u32 {
        for n in 0..16u32 {
            let (lane, j) = (n + 16 * (r & 1), r >> 1);
            if !seen.insert((lane, j)) {
                layout_bad += 1
            }
            let (hi, lr) = (lane >> 4, lane & 15);
            if ar(j, hi) != r || lr != n {
                layout_bad += 1
            }
        }
    }
    if seen.len() != 256 || layout_bad != 0 {
        failures.push(format!(
            "P1 layout: C mapping bijection broken ({} cells, {layout_bad} errors)",
            seen.len()
        ));
    }
    // Scale selector: accumulator j of lane (hi, lr) needs the scale of its C row. BROADCAST16 reads lane
    // 16*hi + selector(j); that lane loaded row scale_row(lane&15, lane>>4).
    let (mut good, mut wrong_mapping_mismatches, mut selector_wrong) = (0u32, 0u32, 0u32);
    for hi in 0..2u32 {
        for j in 0..8u32 {
            let src = 16 * hi + sw(j);
            if sr(src & 15, src >> 4, false) == ar(j, hi) {
                good += 1
            }
            // gfx12 loaded row with the same selector.
            if sr(src & 15, src >> 4, true) != ar(j, hi) {
                wrong_mapping_mismatches += 1
            }
            // selector j (not 2j) with the gfx11 loaded row.
            let s_src = 16 * hi + j;
            if sr(s_src & 15, s_src >> 4, false) != ar(j, hi) {
                selector_wrong += 1
            }
        }
    }
    if good != 16 {
        failures.push(format!(
            "P1 scale selector: only {good}/16 (hi,j) pairs select their C row"
        ))
    }
    if wrong_mapping_mismatches == 0 || selector_wrong == 0 {
        failures.push(
            "P1 scale selector: a deliberately wrong mapping/selector does not mismatch".into(),
        );
    }
    // A weight row (lane lr of both halves) feeds C row r = lr: tile row rb+lr is its own output row.
    json!({
        "helpers_from": if kernel.is_some() { "gfx1100 kernel module" } else { "this file" }, "helper_formula_errors": formula_errors,
        "c_layout": "row 2*j+hi in accumulator j, column (token) lr", "c_cells_checked": 256, "c_layout_errors": layout_bad,
        "scale_row_load": "(lr&14)|hi", "swizzle_broadcast16_selector": "2*j", "hi_j_pairs_correct": good,
        "negative_gfx12_scale_row_mismatching_pairs": wrong_mapping_mismatches,
        "negative_selector_j_mismatching_pairs": selector_wrong,
        "operand_dwords_per_lane_per_epoch": 16, "k16_steps_per_epoch": 8, "wmma_shape": "v_wmma_i32_16x16x16_iu4",
    })
}

fn add64(base: u64, off: u64) -> u64 {
    let (lo, carry) = (base as u32).overflowing_add(off as u32);
    let hi = ((base >> 32) as u32)
        .wrapping_add((off >> 32) as u32)
        .wrapping_add(u32::from(carry));
    (u64::from(hi) << 32) | u64::from(lo)
}

fn prove_add64(failures: &mut Vec<String>) -> Value {
    let bases: [u64; 9] = [
        0,
        0xffff_fff0,
        0x1_0000_0000,
        0x1_ffff_ff00,
        0x2_0000_1000,
        0x7fff_ffff_f000,
        0x0000_7fff_ffff_ffff - 0x4000,
        0x4_0000_0000,
        0xffff_ffff_0000_0000 >> 16,
    ];
    let offsets: [u64; 10] = [
        0,
        1,
        8 * 511,
        0xff0,
        2560 * 48 * 20,
        2560 * 50 * 5,
        0x7fff_ffff,
        0xffff_ffff,
        0x1_0000,
        0x2000_0000,
    ];
    let (mut cases, mut carries, mut truncated_wrong) = (0u64, 0u64, 0u64);
    for &b in &bases {
        for &o in &offsets {
            cases += 1;
            let want = b + o;
            if add64(b, o) != want {
                failures.push(format!("P1 add64: {b:#x} + {o:#x} gave {:#x}", add64(b, o)))
            }
            if (b as u32).checked_add(o as u32).is_none() {
                carries += 1
            }
            // A 32-bit-only add (carry dropped) names the wrong address above 4 GiB.
            let dropped = (b & !0xffff_ffff) | u64::from((b as u32).wrapping_add(o as u32));
            if dropped != want {
                truncated_wrong += 1
            }
        }
    }
    if carries == 0 || truncated_wrong == 0 {
        failures.push("P1 add64: no carry case exercised".into())
    }
    json!({
        "cases": cases, "low_word_carry_cases": carries, "dropped_carry_would_be_wrong_cases": truncated_wrong,
        "bases_above_4gib": bases.iter().filter(|b| **b > 0xffff_ffff).count(),
    })
}

/// Run every P1 obligation. `seed` drives the high-byte / junk generator and the fold witness search.
/// With `kernel`, the gfx1100 module's own CPU mirrors and address helpers are enumerated / cross-checked
/// against the independent references and the address model in this file.
pub fn prove(seed: u64, kernel: Option<&Kernel>) -> Outcome {
    let mut failures = Vec::new();
    let mut rng = Rng(seed);
    let public_w3 = sweep(
        24,
        0xff00_0000,
        &mut rng,
        &|x| w3::spread_model(x),
        &w3_reference,
    );
    let public_w3_json = sweep_json(
        "w3 (public qwen4_moe_sym::w3::spread_model)",
        &public_w3,
        7168,
        &mut failures,
    );
    let network_w2 = sweep(16, 0xffff_0000, &mut rng, &w2_model, &w2_reference);
    let network_w2_json = sweep_json("w2 (independent network)", &network_w2, 3072, &mut failures);
    let mq4_local = prove_mq4(&mut rng, &mq4_model, &mut failures);
    let kernel_models = kernel.map(|k| {
        for (codec, frozen) in [("mq4", 8192u32), ("w2", 3072), ("w3", 7168)] {
            if (k.dot_bound)(codec) != frozen {
                failures.push(format!(
                    "P1 kernel dot_bound({codec}) = {}, frozen {frozen}",
                    (k.dot_bound)(codec)
                ));
            }
        }
        let w3k = sweep(24, 0xff00_0000, &mut rng, k.w3_spread, &w3_reference);
        let w3j = sweep_json("w3 (gfx1100 kernel model)", &w3k, 7168, &mut failures);
        let w2k = sweep(16, 0xffff_0000, &mut rng, k.w2_spread, &w2_reference);
        let w2j = sweep_json("w2 (gfx1100 kernel model)", &w2k, 3072, &mut failures);
        let mq4j = prove_mq4(&mut rng, k.mq4_unpack, &mut failures);
        json!({"w3": w3j, "w2": w2j, "mq4": mq4j})
    });
    let magic = prove_magic(&mut failures);
    let fold = prove_fold(&mut rng, &mut failures);
    let addresses = prove_addresses(&mut failures, kernel);
    let layout = prove_layout(&mut failures, kernel);
    let add = prove_add64(&mut failures);
    let receipt = json!({
        "gate": "B4-P1",
        "scope": "synthetic arithmetic/address model and exhaustive unpack enumeration; no device execution; not G1",
        "seed": seed,
        "kernel_models_cross_checked": kernel.is_some(),
        "w3_unpack": {
            "model": "peacemaker_isa::kernels::qwen4_moe_sym::w3::spread_model",
            "reference": "per-code (2c-7)&15", "junk": "random high byte above bit 23", "result": public_w3_json,
        },
        "w2_unpack": {
            "model": "independent bit-parallel spread network (this file)", "reference": "per-code (2c-3)&15",
            "junk": "random high 16 bits", "result": network_w2_json,
        },
        "mq4_unpack": mq4_local,
        "gfx1100_kernel_models": kernel_models,
        "magic_window": magic,
        "epoch_fold": fold,
        "address_trace": addresses,
        "layout_and_scale_selector": layout,
        "expert_pointer_add64": add,
        "failures": failures.clone(),
    });
    Outcome { receipt, failures }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut seed = 0x4234_5001u64;
    let mut receipt: Option<String> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--seed" => {
                seed = it.next().and_then(|v| v.parse().ok()).unwrap_or_else(|| {
                    eprintln!("--seed N");
                    std::process::exit(2)
                })
            }
            "--receipt" => receipt = it.next().cloned(),
            other => {
                eprintln!("unknown argument {other}");
                std::process::exit(2)
            }
        }
    }
    let out = prove(seed, None);
    let text = serde_json::to_string_pretty(&out.receipt).expect("receipt serialises") + "\n";
    match receipt {
        Some(path) => {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .unwrap_or_else(|e| {
                    eprintln!("{path}: {e} (existing receipts are never overwritten)");
                    std::process::exit(2)
                });
            f.write_all(text.as_bytes()).unwrap_or_else(|e| {
                eprintln!("{path}: {e}");
                std::process::exit(2)
            });
        }
        None => print!("{text}"),
    }
    for f in &out.failures {
        eprintln!("FAIL {f}")
    }
    std::process::exit(i32::from(!out.failures.is_empty()));
}
