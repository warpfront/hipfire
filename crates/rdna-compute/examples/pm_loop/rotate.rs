// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! `moe_rotate_i4_pm` workload: the certified gfx1201 PeaceMaker twin of the
//! down A4 producer `qwen4_moe_rotate128_i4` (`peacemaker-isa emit --kernel
//! moe_rotate_i4_pm`): grouped BF16 SwiGLU rows `h [P, K]` -> FWHT128 (signs1,
//! `1/sqrt(128)`, signs2) -> `block_i4_128` under the frozen C2 `{5,7}`
//! candidates, one 72-byte record per (flat slot `sorted[p]`, K128 group `g`)
//! at `xq + 72 * (g * L + sorted[p])`; a pad row (`sorted[p] < 0`) writes
//! nothing.
//!
//! # Knobs
//!
//! * `--variant rows|gp` (default `rows`): `rows` is the flat-slot sidecar
//!   producer (module and symbol `moe_rotate_i4_pm_gfx1201`, shipped
//!   `kernels/moe_rotate_i4_pm_gfx1201.hxaco`); `gp` is the grouped-position
//!   twin (`moe_rotate_i4_gp_pm_gfx1201`, record `g * P + row`, exported by
//!   module `qwen4_moe_down_gp_pm_gfx1201` next to the two down GEMM entries,
//!   shipped `kernels/qwen4_moe_down_gp_pm_gfx1201.hxaco`). The candidate and
//!   the shipped baseline use the matching module and symbol; the ABI, grid and
//!   the record bytes are the same.
//! * `--epi all|r128` (default `all`, as the CLI): `all` links the whole
//!   module text, `r128` the single `moe_rotate_i4_pm` entry (rows only; the
//!   gp module has no per-entry text).
//! * `--ref-rows full|sample:<n>`: `full` checks every output byte (unwritten
//!   records must stay zero); `sample:<n>` checks `n` evenly spaced valid grouped
//!   rows, every K128 group of each. Default `full` while the valid
//!   records `L * K/128` are at most 2^18, else `sample:4096`.
//!
//! # Shape grammar
//!
//! `[fn:t<T>|a3b:t<T>][,key=value]...` (empty = `fn:t4096`). `T` prefill tokens.
//! The preset fixes the MoE dims, the keys override them:
//!
//! * `fn:t<T>` Flash-Next: `E=512,TOP=10,K=640` (hidden 2560, per-expert
//!   intermediate 640; `examples/pm_loop/moe.rs:213` and `moe.rs:29-31`,
//!   `examples/qwen4_moe_sym.rs:96 GEO_FN`). `K` is the down GEMM's K, the
//!   producer's row width (`d.mi` at `hipfire-dispatch/src/pipeline/slowlink_prefill.rs:1979`,
//!   `p.down_k` at `qt44_qt53_prefill.rs:1820`, `plan.intermediate` at
//!   `pipeline/mod.rs:7610`).
//! * `a3b:t<T>` A3B: `E=256,TOP=8,K=512` (`moe.rs:214`, `qwen4_moe_sym.rs:97
//!   GEO_A3B`; the `K = 512` down input of the launcher docs,
//!   `rdna-compute/src/moe_rotate_i4_pm.rs:80`). The oracle's row counts
//!   (`examples/moe_rotate_i4_pm_oracle.rs:67`) are `t512`, `t4096`, `t8192`.
//! * keys `T`, `TOP`, `E`, `K` (a multiple of 128), `seed`.
//!
//! The launch geometry is the pad16 grouping of the production routing
//! (`moe.rs:644-660`, `examples/qwen4_moe_sym.rs:395-428`): `L = T * TOP` flat
//! slots (`total_slots = batch_size * k_top`, `qt44_qt53_prefill.rs:1816`), `TOP`
//! distinct uniform experts per token, each expert's run stable-sorted and
//! padded to a multiple of 16 with `-1`, `P = ceil((L + 15 E) / 16) * 16`
//! grouped rows (`sorted[p] = -1` tail included).
//!
//! # Reference
//!
//! Exact: the CPU twin of the record arithmetic transcribed from
//! `crates/peacemaker-isa/tests/moe_rotate_i4_pm.rs:490-576` (FWHT128 with the
//! sign tables the kernel reads, f32 `1/sqrt(128)`, `fake_quantize_block_i4_128`
//! with two candidates).

use crate::common::{
    bf16_normal_bytes, emitted_object, emu_run, no_leftover, put_ptr, put_u32, read_rip, refused,
    rip_object, shipped_object, take_dim, Elem, EmuBuf, RipSource, ShapeSpec, Xs64,
};
use crate::prod::{self, Covered, Expect, OutBuf, Producer, Ptrs};
use crate::workload::{Knobs, Launch, Object, Result, Workload};
use peacemaker_isa::kernels::{moe_rotate_i4_pm, mq_i4_a4, qwen4_moe_down_gp};
use peacemaker_isa::Arch;
use rayon::prelude::*;
use rdna_compute::moe_rotate_i4_pm::{
    validate_moe_rotate_i4_gp_pm, validate_moe_rotate_i4_pm, MoeRotateI4PmGeometry,
};
use serde_json::{json, Value};

const ARCH: Arch = Arch::Gfx1201;
/// Rotation width and A4 group width in columns.
pub(crate) const GROUP: usize = moe_rotate_i4_pm::GROUP as usize;
/// Bytes of one `block_i4_128` record.
pub(crate) const REC: usize = moe_rotate_i4_pm::RECORD_BYTES as usize;
/// `Gpu::qwen4_moe_rotate128_i4_pm` block (`MOE_ROTATE_I4_PM_BLOCK`, `rdna-compute/src/moe_rotate_i4_pm.rs:40`).
const BLOCK: u32 = 32;
/// Largest valid-record count checked in full by default.
const FULL_REF_RECORDS: usize = 1 << 18;
const DEFAULT_SAMPLE_ROWS: usize = 4096;
const REFERENCE_NAME: &str = "cpu_fwht128_i4_c2";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Variant {
    Rows,
    Gp,
}

impl Variant {
    fn parse(s: &str) -> Result<Self> {
        match s {
            "rows" => Ok(Self::Rows),
            "gp" => Ok(Self::Gp),
            _ => Err(format!("--variant {s}: expected rows|gp")),
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::Rows => "rows",
            Self::Gp => "gp",
        }
    }
    /// The launched symbol.
    fn symbol(self) -> String {
        match self {
            Self::Rows => moe_rotate_i4_pm::Spec::module(ARCH),
            Self::Gp => moe_rotate_i4_pm::symbol_gp(ARCH),
        }
    }
}

#[derive(Clone, Debug)]
struct Shape {
    /// `fn:t4096`, `a3b:t512`, or `fn:t4096(default)`.
    name: String,
    tokens: usize,
    top: usize,
    experts: usize,
    k: usize,
    seed: usize,
}

fn parse_shape(s: &str) -> Result<Shape> {
    let spec = ShapeSpec::parse(s)?;
    let (name, defaults) = match &spec.preset {
        None => (
            "fn:t4096(default)".to_owned(),
            [
                ("T", 4096),
                ("TOP", 10),
                ("E", 512),
                ("K", 640),
                ("seed", 1),
            ],
        ),
        Some((fam, n)) => {
            let t = n
                .strip_prefix('t')
                .and_then(|d| d.parse::<usize>().ok())
                .ok_or_else(|| format!("shape preset {fam}:{n}: expected {fam}:t<tokens>"))?;
            let (top, e, k) = match fam.as_str() {
                "fn" => (10, 512, 640),
                "a3b" => (8, 256, 512),
                other => return Err(format!("shape preset family {other} (fn|a3b)")),
            };
            (
                format!("{fam}:{n}"),
                [("T", t), ("TOP", top), ("E", e), ("K", k), ("seed", 1)],
            )
        }
    };
    let mut m = spec.resolve(&defaults);
    let shape = Shape {
        name,
        tokens: take_dim(&mut m, "T")?,
        top: take_dim(&mut m, "TOP")?,
        experts: take_dim(&mut m, "E")?,
        k: take_dim(&mut m, "K")?,
        seed: take_dim(&mut m, "seed")?,
    };
    no_leftover(&m)?;
    if shape.tokens == 0 || shape.top == 0 || shape.top > shape.experts {
        return Err(format!(
            "shape: need T > 0 and 0 < TOP <= E (T={}, TOP={}, E={})",
            shape.tokens, shape.top, shape.experts
        ));
    }
    Ok(shape)
}

/// `TOP` distinct uniform experts per token, stable-grouped by expert with each
/// run padded to 16 (`pm_loop/moe.rs:644-660`); `-1` fills the tail to `p`.
fn grouped_sorted(s: &Shape, p: usize) -> Vec<i32> {
    let mut rng = Xs64::new(s.seed as u64 ^ 0x726f_7461_7465);
    let mut route = Vec::with_capacity(s.tokens * s.top);
    for _ in 0..s.tokens {
        let start = route.len();
        while route.len() - start < s.top {
            let e = (rng.next() % s.experts as u64) as u32;
            if !route[start..].contains(&e) {
                route.push(e);
            }
        }
    }
    let mut counts = vec![0usize; s.experts];
    for &e in &route {
        counts[e as usize] += 1;
    }
    let mut fill = Vec::with_capacity(s.experts);
    let mut at = 0;
    for c in &counts {
        fill.push(at);
        at += c.div_ceil(16) * 16;
    }
    let mut sorted = vec![-1i32; p];
    for (f, &e) in route.iter().enumerate() {
        sorted[fill[e as usize]] = f as i32;
        fill[e as usize] += 1;
    }
    sorted
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RefRows {
    Full,
    Sample(usize),
}

impl RefRows {
    fn parse(knob: Option<&str>, valid_records: usize) -> Result<Self> {
        match knob {
            None if valid_records <= FULL_REF_RECORDS => Ok(Self::Full),
            None => Ok(Self::Sample(DEFAULT_SAMPLE_ROWS)),
            Some("full") => Ok(Self::Full),
            Some(s) => s
                .strip_prefix("sample:")
                .and_then(|n| n.parse().ok())
                .filter(|&n| n > 0)
                .map(Self::Sample)
                .ok_or_else(|| format!("--ref-rows {s}: expected full|sample:<n>")),
        }
    }
    fn label(self) -> String {
        match self {
            Self::Full => "full".into(),
            Self::Sample(n) => format!("sample:{n}"),
        }
    }
}

struct Rotate {
    variant: Variant,
    /// `--epi r128`: link the single entry instead of the module.
    single: bool,
    shape: Shape,
    ref_rows: RefRows,
    /// Grouped rows `P`.
    p: usize,
    /// Flat slots `L`.
    l: usize,
    geo: MoeRotateI4PmGeometry,
    sorted: Vec<i32>,
    /// `--rip`: the candidate is this script's entry, not the Rust emit.
    rip: RipSource,
}

#[cfg(test)]
pub fn new(shape: &str, knobs: &mut Knobs) -> Result<Box<dyn Workload>> {
    new_with_rip(shape, knobs, None)
}

pub fn new_with_rip(
    shape: &str,
    knobs: &mut Knobs,
    rip: Option<&std::path::Path>,
) -> Result<Box<dyn Workload>> {
    let variant = Variant::parse(&knobs.take_or("variant", "rows"))?;
    let single = match (knobs.take_or("epi", "all").as_str(), variant) {
        ("all", _) => false,
        ("r128", Variant::Rows) => true,
        (other, _) => {
            return Err(format!(
                "--epi {other} (all|r128; r128 only with --variant rows)"
            ))
        }
    };
    let ref_rows = knobs.take("ref-rows");
    let mut rotate = Rotate::build(variant, single, parse_shape(shape)?, ref_rows.as_deref())?;
    rotate.rip = read_rip(rip)?;
    Ok(prod::boxed(rotate))
}

impl Rotate {
    fn build(variant: Variant, single: bool, shape: Shape, ref_rows: Option<&str>) -> Result<Self> {
        let l = shape.tokens * shape.top;
        let p = (l + shape.experts * 15).div_ceil(16) * 16;
        let geo = match variant {
            Variant::Rows => validate_moe_rotate_i4_pm(shape.k, p, l)?,
            Variant::Gp => validate_moe_rotate_i4_gp_pm(shape.k, p, l)?,
        };
        let ref_rows = RefRows::parse(ref_rows, l * geo.groups)?;
        let sorted = grouped_sorted(&shape, p);
        Ok(Self {
            variant,
            single,
            shape,
            ref_rows,
            p,
            l,
            geo,
            sorted,
            rip: None,
        })
    }

    fn valid_rows(&self) -> Vec<usize> {
        (0..self.p).filter(|&r| self.sorted[r] >= 0).collect()
    }

    /// Record index of (group `g`, grouped row `row`) (`row` valid).
    fn record_index(&self, g: usize, row: usize) -> usize {
        match self.variant {
            Variant::Rows => g * self.l + self.sorted[row] as usize,
            Variant::Gp => g * self.p + row,
        }
    }

    fn input_sizes(&self) -> [(&'static str, usize); 4] {
        [
            ("h", self.p * self.shape.k * 2),
            ("sorted", self.p * 4),
            ("signs1", GROUP * 4),
            ("signs2", GROUP * 4),
        ]
    }

    fn knobs_value(&self) -> Value {
        json!({
            "variant": self.variant.name(),
            "epi": if self.single { "r128" } else { "all" },
            "export": self.variant.symbol(),
        })
    }

    fn run_emu(&self) -> Result<Value> {
        // 2 experts x 6 tokens: two padded runs of 6 + a pad tail, 2 K128 groups.
        let mut small = Self::build(
            self.variant,
            self.single,
            Shape {
                name: "emu".into(),
                tokens: 6,
                top: 2,
                experts: 2,
                k: 256,
                seed: self.shape.seed,
            },
            Some("full"),
        )?;
        small.rip = self.rip.clone();
        let obj = small.candidate()?;
        let inputs = small.inputs();
        let outs = small.outputs();
        let [h, s, s1, s2, xq] = [
            0x2000_0000u64,
            0x3000_0000,
            0x4000_0000,
            0x5000_0000,
            0x6000_0000,
        ];
        let launch = small.launch(&Ptrs {
            inputs: vec![h, s, s1, s2],
            outputs: vec![xq],
        });
        let names = ["h", "sorted", "signs1", "signs2"];
        let mut bufs: Vec<EmuBuf> = names
            .into_iter()
            .zip([h, s, s1, s2])
            .zip(inputs.iter().cloned())
            .map(|((name, va), bytes)| EmuBuf { name, va, bytes })
            .collect();
        bufs.push(EmuBuf {
            name: "xq",
            va: xq,
            bytes: outs[0].init.clone(),
        });
        let mem = emu_run(
            &obj.bytes,
            &obj.launch_symbol,
            launch.kernarg.clone(),
            bufs,
            launch.grid,
            launch.block,
            launch.lds,
        )?;
        let got = mem.read(xq, outs[0].init.len())?;
        let expect = small.reference(&inputs, &outs)?;
        let want = &expect.outputs[0]
            .as_ref()
            .ok_or("no reference bytes")?
            .bytes;
        let bad = got.iter().zip(want).filter(|(a, b)| a != b).count();
        Ok(json!({
            "ran": true,
            "bit_exact": bad == 0,
            "shape": {
                "tokens": small.shape.tokens, "top_k": small.shape.top,
                "experts": small.shape.experts, "K": small.shape.k,
                "P": small.p, "L": small.l, "grid": launch.grid,
            },
            "symbol": obj.launch_symbol,
            "reference": expect.reference,
            "mismatching_elements": bad,
        }))
    }
}

impl Producer for Rotate {
    fn kernel(&self) -> &'static str {
        "moe_rotate_i4_pm"
    }

    fn shape_json(&self) -> Value {
        let s = &self.shape;
        json!({
            "preset": s.name, "tokens": s.tokens, "top_k": s.top, "experts": s.experts,
            "K": s.k, "groups": self.geo.groups, "P": self.p, "L": self.l,
            "pad_rows": self.p - self.l, "seed": s.seed,
            "grid": self.geo.grid, "block": BLOCK, "ref_rows": self.ref_rows.label(),
            "buffers": prod::buffers_json(&self.input_sizes(), &self.outputs()),
        })
    }

    fn knobs_json(&self) -> Value {
        self.knobs_value()
    }

    fn candidate(&self) -> Result<Object> {
        if let Some(rip) = &self.rip {
            let spec = moe_rotate_i4_pm::Spec { arch: ARCH };
            let (variant, generator) = match self.variant {
                Variant::Rows => ("rows", moe_rotate_i4_pm::emit(spec)?),
                Variant::Gp => ("gp", moe_rotate_i4_pm::emit_gp(spec)?),
            };
            return rip_object(
                rip,
                ARCH,
                variant,
                &self.variant.symbol(),
                &generator.s_text,
                self.knobs_value(),
            );
        }
        let (text, exports, what) = match (self.variant, self.single) {
            (Variant::Rows, true) => (
                moe_rotate_i4_pm::emit(moe_rotate_i4_pm::Spec { arch: ARCH })?.s_text,
                vec![self.variant.symbol()],
                "peacemaker_isa::kernels::moe_rotate_i4_pm::emit + native::assemble",
            ),
            (Variant::Rows, false) => (
                moe_rotate_i4_pm::emit_module(ARCH)?.1,
                vec![self.variant.symbol()],
                "peacemaker_isa::kernels::moe_rotate_i4_pm::emit_module + native::assemble",
            ),
            (Variant::Gp, _) => (
                qwen4_moe_down_gp::emit_module(ARCH)?.1,
                qwen4_moe_down_gp::symbols(ARCH).to_vec(),
                "peacemaker_isa::kernels::qwen4_moe_down_gp::emit_module + native::assemble",
            ),
        };
        emitted_object(
            &text,
            ARCH,
            &self.variant.symbol(),
            &exports,
            what,
            self.knobs_value(),
        )
    }

    fn baseline(&self) -> Result<Object> {
        let symbol = self.variant.symbol();
        let knobs = json!({"variant": self.variant.name()});
        match self.variant {
            Variant::Rows => shipped_object(
                rdna_compute::kernels::MOE_ROTATE_I4_PM_GFX1201,
                "kernels/moe_rotate_i4_pm_gfx1201.hxaco",
                "crates/rdna-compute/src/moe_rotate_i4_pm.rs:54-61 (moe_rotate_i4_pm_kernel), :390 ensure_embedded_kernel, :414 launch",
                &symbol,
                &[&symbol],
                knobs,
            ),
            Variant::Gp => {
                let symbols = qwen4_moe_down_gp::symbols(ARCH);
                let exports: Vec<&str> = symbols.iter().map(String::as_str).collect();
                shipped_object(
                    rdna_compute::kernels::QWEN4_MOE_DOWN_GP_PM_GFX1201,
                    "kernels/qwen4_moe_down_gp_pm_gfx1201.hxaco",
                    "crates/rdna-compute/src/moe_rotate_i4_pm.rs:290-295 ensure_embedded_kernel (QWEN4_MOE_DOWN_GP_PM_GFX1201), :319 launch",
                    &symbol,
                    &exports,
                    knobs,
                )
            }
        }
    }

    fn inputs(&self) -> Vec<Vec<u8>> {
        let f32_bytes =
            |v: Vec<f32>| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        vec![
            // Grouped BF16 SwiGLU rows (pad rows carry data too: the kernel must not read them).
            bf16_normal_bytes(self.shape.seed as u64 ^ 0x68, self.p * self.shape.k, 0.5),
            self.sorted.iter().flat_map(|s| s.to_le_bytes()).collect(),
            // `Gpu::ensure_mq_signs_128` (`rdna-compute/src/scratch.rs:2090-2091`).
            f32_bytes(rdna_compute::gen_fwht_signs(mq_i4_a4::SIGN_SEEDS.0, GROUP)),
            f32_bytes(rdna_compute::gen_fwht_signs(mq_i4_a4::SIGN_SEEDS.1, GROUP)),
        ]
    }

    fn outputs(&self) -> Vec<OutBuf> {
        vec![OutBuf::zeros("xq", self.geo.xq_bytes, Elem::U8)]
    }

    /// `Gpu::qwen4_moe_rotate128_i4_pm` / `_gp_pm` (`rdna-compute/src/moe_rotate_i4_pm.rs:398-431`,
    /// `:303-336`): grid `[P, K/128, 1]`, block 32, no LDS, 56-byte kernarg
    /// `h@0 sorted@8 xq@16 signs1@24 signs2@32 K@40 P@44 L@48`.
    fn launch(&self, p: &Ptrs) -> Launch {
        let mut k = Vec::with_capacity(moe_rotate_i4_pm::KERNARG_BYTES as usize);
        for ptr in [
            p.inputs[0],
            p.inputs[1],
            p.outputs[0],
            p.inputs[2],
            p.inputs[3],
        ] {
            put_ptr(&mut k, ptr);
        }
        for v in [self.shape.k, self.p, self.l] {
            put_u32(&mut k, v as u32);
        }
        Launch {
            kernarg: k,
            grid: self.geo.grid,
            block: [BLOCK, 1, 1],
            lds: 0,
        }
    }

    fn reference(&self, inputs: &[Vec<u8>], outputs: &[OutBuf]) -> Result<Expect> {
        let (h, s1, s2) = (
            &inputs[0],
            parse_signs(&inputs[2])?,
            parse_signs(&inputs[3])?,
        );
        let valid = self.valid_rows();
        let rows: Vec<usize> = match self.ref_rows {
            RefRows::Sample(n) if n < valid.len() => {
                (0..n).map(|i| valid[i * valid.len() / n]).collect()
            }
            _ => valid,
        };
        let (k, groups) = (self.shape.k, self.geo.groups);
        let records: Vec<(usize, [u8; REC])> = rows
            .par_iter()
            .flat_map_iter(|&row| {
                let (s1, s2) = (&s1, &s2);
                (0..groups).map(move |g| {
                    let off = (row * k + g * GROUP) * 2;
                    let mut grp = [0u16; GROUP];
                    for (v, c) in grp.iter_mut().zip(h[off..off + GROUP * 2].chunks_exact(2)) {
                        *v = u16::from_le_bytes([c[0], c[1]]);
                    }
                    (self.record_index(g, row), quantize(&rotate(&grp, s1, s2)))
                })
            })
            .collect();
        let mut bytes = vec![0u8; outputs[0].init.len()];
        let mut ranges = Vec::with_capacity(records.len());
        for (idx, rec) in &records {
            bytes[idx * REC..(idx + 1) * REC].copy_from_slice(rec);
            ranges.push(idx * REC..(idx + 1) * REC);
        }
        let covered = match self.ref_rows {
            RefRows::Full => Covered::full(bytes),
            RefRows::Sample(_) => Covered { bytes, ranges },
        };
        Ok(Expect {
            outputs: vec![Some(covered)],
            reference: REFERENCE_NAME.into(),
        })
    }

    fn emu(&self) -> Value {
        match self.run_emu() {
            Ok(v) => v,
            Err(e) => refused(e.split(" | ").next().unwrap_or(&e)),
        }
    }
}

fn parse_signs(b: &[u8]) -> Result<[f32; GROUP]> {
    if b.len() != GROUP * 4 {
        return Err(format!(
            "sign table is {} bytes, expected {}",
            b.len(),
            GROUP * 4
        ));
    }
    let mut out = [0f32; GROUP];
    for (s, c) in out.iter_mut().zip(b.chunks_exact(4)) {
        *s = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
    }
    Ok(out)
}

/// FWHT128 as the kernel: widen, `* signs1`, butterflies of strides 1..64
/// (`lo' = lo + hi`, `hi' = lo - hi`), `(v * 0.0883883476) * signs2`
/// (`crates/peacemaker-isa/tests/moe_rotate_i4_pm.rs:490-511`).
pub(crate) fn rotate(h: &[u16; GROUP], s1: &[f32; GROUP], s2: &[f32; GROUP]) -> [f32; GROUP] {
    let mut v = [0f32; GROUP];
    for i in 0..GROUP {
        v[i] = f32::from_bits(u32::from(h[i]) << 16) * s1[i];
    }
    let mut stride = 1;
    while stride < GROUP {
        for base in (0..GROUP).step_by(2 * stride) {
            for j in base..base + stride {
                let (lo, hi) = (v[j], v[j + stride]);
                v[j] = lo + hi;
                v[j + stride] = lo - hi;
            }
        }
        stride *= 2;
    }
    let scale = f32::from_bits(mq_i4_a4::ROTATION_SCALE_BITS);
    for i in 0..GROUP {
        v[i] = (v[i] * scale) * s2[i];
    }
    v
}

/// The 72-byte `block_i4_128` record of one rotated group:
/// `fake_quantize_block_i4_128(candidates = 2)` (`hipfire-quantize/src/calibration.rs`),
/// four values per simulated lane and the XOR 16,8,4,2,1 trees
/// (`crates/peacemaker-isa/tests/moe_rotate_i4_pm.rs:515-576`).
pub(crate) fn quantize(x: &[f32; GROUP]) -> [u8; REC] {
    let mut lane_amax = [0f32; 32];
    for (lane, a) in lane_amax.iter_mut().enumerate() {
        let b = lane * 4;
        *a = x[b]
            .abs()
            .max(x[b + 1].abs())
            .max(x[b + 2].abs())
            .max(x[b + 3].abs());
    }
    for off in [16usize, 8, 4, 2, 1] {
        let prior = lane_amax;
        for lane in 0..32 {
            lane_amax[lane] = lane_amax[lane].max(prior[lane ^ off]);
        }
    }
    let amax = lane_amax[0];
    let mut best_d = 1.0f32;
    if amax != 0.0 {
        let base = (amax / 7.0f32) * 0.5f32;
        let mut best_mse = 1.0e30f32;
        for mult in [f32::from_bits(mq_i4_a4::CAND0_MULT_BITS), 2.0f32] {
            let d = base * mult;
            let mut lane_mse = [0f32; 32];
            for (lane, m) in lane_mse.iter_mut().enumerate() {
                let mut mse = 0f32;
                for e in 0..4 {
                    let v = x[lane * 4 + e];
                    let q = (v / d).round_ties_even().clamp(-8.0, 7.0);
                    let err = (-q).mul_add(d, v);
                    mse = err.mul_add(err, mse);
                }
                *m = mse;
            }
            for off in [16usize, 8, 4, 2, 1] {
                let prior = lane_mse;
                for lane in 0..32 {
                    lane_mse[lane] += prior[lane ^ off];
                }
            }
            if lane_mse[0] < best_mse {
                best_mse = lane_mse[0];
                best_d = d;
            }
        }
    }
    let mut q = [0i32; GROUP];
    if amax != 0.0 {
        for (qi, v) in q.iter_mut().zip(x) {
            *qi = (v / best_d).round_ties_even().clamp(-8.0, 7.0) as i32;
        }
    }
    let mut rec = [0u8; REC];
    rec[0..4].copy_from_slice(&best_d.to_le_bytes());
    rec[4..8].copy_from_slice(&q.iter().sum::<i32>().to_le_bytes());
    for i in 0..GROUP / 2 {
        rec[8 + i] = ((q[2 * i] & 15) | ((q[2 * i + 1] & 15) << 4)) as u8;
    }
    rec
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::sha256_hex;

    fn knobs(kv: &[(&str, &str)]) -> Knobs {
        Knobs(kv.iter().map(|(k, v)| ((*k).into(), (*v).into())).collect())
    }

    fn build(variant: Variant, single: bool, shape: &str) -> Rotate {
        Rotate::build(variant, single, parse_shape(shape).unwrap(), None).unwrap()
    }

    #[test]
    fn shape_grammar_and_presets() {
        let d = parse_shape("").unwrap();
        assert_eq!((d.tokens, d.top, d.experts, d.k), (4096, 10, 512, 640));
        let a = parse_shape("a3b:t512").unwrap();
        assert_eq!((a.tokens, a.top, a.experts, a.k), (512, 8, 256, 512));
        let o = parse_shape("fn:t16,K=256,E=8,TOP=2,seed=3").unwrap();
        assert_eq!(
            (o.tokens, o.top, o.experts, o.k, o.seed),
            (16, 2, 8, 256, 3)
        );
        assert_eq!(parse_shape("T=7").unwrap().tokens, 7);
        for bad in [
            "fn:4096",
            "a3b:x",
            "fn:t0",
            "TOP=600",
            "foo=1",
            "fn:t8,fn:t9",
        ] {
            assert!(parse_shape(bad).is_err(), "{bad}");
        }
        assert!(Rotate::build(Variant::Rows, false, parse_shape("K=100").unwrap(), None).is_err());
        // Main prefill shape: L = 40960 slots, P = ceil((L + 15 * 512) / 16) * 16 grouped rows.
        let w = build(Variant::Rows, false, "fn:t4096");
        assert_eq!(
            (w.l, w.p),
            (40960, (40960 + 512 * 15usize).div_ceil(16) * 16)
        );
        assert_eq!(w.geo.grid, [w.p as u32, 5, 1]);
        assert_eq!(w.geo.xq_bytes, 5 * w.l * REC);
        assert_eq!(build(Variant::Gp, false, "a3b:t8192").geo.xq_bytes % REC, 0);
    }

    #[test]
    fn grouping_is_a_padded_permutation() {
        let w = build(Variant::Rows, false, "T=100,TOP=4,E=16,K=128");
        assert_eq!(w.sorted.len(), w.p);
        let mut seen = vec![false; w.l];
        for &s in w.sorted.iter().filter(|&&s| s >= 0) {
            assert!(
                !std::mem::replace(&mut seen[s as usize], true),
                "slot {s} twice"
            );
        }
        assert!(seen.iter().all(|&b| b), "every slot appears");
        assert!(w.sorted.iter().any(|&s| s < 0), "pad rows exist");
        assert_eq!(w.valid_rows().len(), w.l);
    }

    #[test]
    fn kernarg_layout_and_grid() {
        let w = build(Variant::Rows, false, "fn:t16");
        let l = w.launch(&Ptrs {
            inputs: vec![0x10, 0x20, 0x30, 0x40],
            outputs: vec![0x50],
        });
        assert_eq!(l.kernarg.len(), 52, "explicit bytes: 5 pointers + K, P, L");
        let u64_at = |o: usize| u64::from_le_bytes(l.kernarg[o..o + 8].try_into().unwrap());
        let u32_at = |o: usize| u32::from_le_bytes(l.kernarg[o..o + 4].try_into().unwrap());
        assert_eq!(
            [u64_at(0), u64_at(8), u64_at(16), u64_at(24), u64_at(32)],
            [0x10, 0x20, 0x50, 0x30, 0x40]
        );
        assert_eq!(
            [u32_at(40), u32_at(44), u32_at(48)],
            [640, w.p as u32, w.l as u32]
        );
        assert_eq!(
            (l.grid, l.block, l.lds),
            ([w.p as u32, 5, 1], [32, 1, 1], 0)
        );
    }

    #[test]
    fn knobs_select_module_and_symbol() {
        let r = new("fn:t16", &mut knobs(&[])).unwrap();
        assert_eq!(r.candidate_knobs()["export"], "moe_rotate_i4_pm_gfx1201");
        let g = new("fn:t16", &mut knobs(&[("variant", "gp")])).unwrap();
        assert_eq!(g.candidate_knobs()["export"], "moe_rotate_i4_gp_pm_gfx1201");
        let og = g.candidate().unwrap();
        assert_eq!(og.launch_symbol, "moe_rotate_i4_gp_pm_gfx1201");
        assert_eq!(
            og.exports.len(),
            3,
            "the gp module exports the producer and both down entries"
        );
        assert!(og.exports.contains(&og.launch_symbol));
        assert!(new("fn:t16", &mut knobs(&[("variant", "x")])).is_err());
        assert!(new("fn:t16", &mut knobs(&[("variant", "gp"), ("epi", "r128")])).is_err());
        assert!(new("fn:t16", &mut knobs(&[("epi", "nope")])).is_err());
        assert!(new("fn:t16", &mut knobs(&[("ref-rows", "sample:0")])).is_err());
    }

    #[test]
    fn candidate_equals_fresh_emit_and_assemble() {
        let link = |text: &str| peacemaker_isa::native::assemble(text, ARCH).unwrap();
        let module = build(Variant::Rows, false, "fn:t16").candidate().unwrap();
        assert_eq!(
            module.bytes,
            link(&moe_rotate_i4_pm::emit_module(ARCH).unwrap().1)
        );
        assert_eq!(module.exports, vec![module.launch_symbol.clone()]);
        let single = build(Variant::Rows, true, "fn:t16").candidate().unwrap();
        let entry = moe_rotate_i4_pm::emit(moe_rotate_i4_pm::Spec { arch: ARCH }).unwrap();
        assert_eq!(single.bytes, link(&entry.s_text));
        let gp = build(Variant::Gp, false, "fn:t16").candidate().unwrap();
        assert_eq!(
            gp.bytes,
            link(&qwen4_moe_down_gp::emit_module(ARCH).unwrap().1)
        );
    }

    #[test]
    fn baseline_is_the_shipped_bundle_with_its_symbols() {
        let r = build(Variant::Rows, false, "fn:t16").baseline().unwrap();
        assert_eq!(r.bytes, rdna_compute::kernels::MOE_ROTATE_I4_PM_GFX1201);
        assert_eq!(r.launch_symbol, "moe_rotate_i4_pm_gfx1201");
        assert_eq!(r.source["sha256"], sha256_hex(&r.bytes));
        let g = build(Variant::Gp, false, "fn:t16").baseline().unwrap();
        assert_eq!(g.bytes, rdna_compute::kernels::QWEN4_MOE_DOWN_GP_PM_GFX1201);
        assert_eq!(g.launch_symbol, "moe_rotate_i4_gp_pm_gfx1201");
        assert_eq!(g.exports, qwen4_moe_down_gp::symbols(ARCH).to_vec());
        assert_eq!(g.source["sha256"], sha256_hex(&g.bytes));
        assert!(g.source["source"]
            .as_str()
            .unwrap()
            .contains("moe_rotate_i4_pm.rs:290"));
    }

    /// One valid row, `h = e0` (bf16 1.0 at column 0): the rotation is all
    /// `s1[0] * 0.0883883476 * s2[i]`, the `{5,7}` winner is `d = (amax / 7 / 2) * 2`
    /// (candidate 1) and every code is `7 * s1[0] * s2[i]`.
    #[test]
    fn tiny_reference_is_hand_checkable() {
        let w = build(Variant::Rows, false, "T=1,TOP=1,E=1,K=128");
        assert_eq!((w.p, w.l), (16, 1));
        assert_eq!(w.sorted[0], 0);
        let mut inputs = w.inputs();
        inputs[0] = vec![0; w.p * 128 * 2];
        inputs[0][0..2].copy_from_slice(&0x3f80u16.to_le_bytes());
        let outs = w.outputs();
        let exp = w.reference(&inputs, &outs).unwrap();
        assert_eq!(exp.reference, REFERENCE_NAME);
        let rec = &exp.outputs[0].as_ref().unwrap().bytes;
        assert_eq!(rec.len(), REC, "one record: pad rows write nothing");
        let (s1, s2) = (
            parse_signs(&inputs[2]).unwrap(),
            parse_signs(&inputs[3]).unwrap(),
        );
        let amax = f32::from_bits(mq_i4_a4::ROTATION_SCALE_BITS);
        assert_eq!(
            f32::from_le_bytes(rec[0..4].try_into().unwrap()),
            (amax / 7.0 * 0.5) * 2.0
        );
        let q: Vec<i32> = s2
            .iter()
            .map(|s| (7.0 * s1[0] * s).round() as i32)
            .collect();
        assert_eq!(
            i32::from_le_bytes(rec[4..8].try_into().unwrap()),
            q.iter().sum::<i32>()
        );
        for j in 0..64 {
            assert_eq!(
                rec[8 + j],
                ((q[2 * j] & 15) | ((q[2 * j + 1] & 15) << 4)) as u8,
                "byte {j}"
            );
        }
        // An all-zero group takes the unit-block path: d = 1, every code 0.
        inputs[0].fill(0);
        let z = w.reference(&inputs, &outs).unwrap();
        let rec = &z.outputs[0].as_ref().unwrap().bytes;
        assert_eq!(&rec[0..4], &1.0f32.to_le_bytes());
        assert!(rec[4..].iter().all(|&b| b == 0));
    }

    #[test]
    fn reference_follows_each_variants_record_index() {
        for variant in [Variant::Rows, Variant::Gp] {
            let w = build(variant, false, "T=3,TOP=2,E=2,K=256");
            let inputs = w.inputs();
            let outs = w.outputs();
            let exp = w.reference(&inputs, &outs).unwrap();
            let bytes = &exp.outputs[0].as_ref().unwrap().bytes;
            let live = (0..bytes.len() / REC)
                .filter(|i| bytes[i * REC..(i + 1) * REC].iter().any(|&b| b != 0))
                .count();
            assert_eq!(
                live,
                w.l * w.geo.groups,
                "{variant:?}: every valid (row, group) record, nothing else"
            );
        }
        let w = Rotate::build(
            Variant::Rows,
            false,
            parse_shape("T=40,TOP=2,E=4,K=128").unwrap(),
            Some("sample:5"),
        )
        .unwrap();
        let exp = w.reference(&w.inputs(), &w.outputs()).unwrap();
        assert_eq!(exp.outputs[0].as_ref().unwrap().ranges.len(), 5);
    }

    const TWIN: &str = include_str!("../../../peacemaker-isa/src/kernels/moe_rotate_i4_pm.rip");

    #[test]
    fn rip_twin_is_the_candidate_of_both_variants_and_identical_to_the_generator() {
        for (variant, name) in [(Variant::Rows, "rows"), (Variant::Gp, "gp")] {
            let mut w = build(variant, false, "fn:t16");
            w.rip = Some(("twin.rip".into(), TWIN.into()));
            let o = w.candidate().unwrap();
            assert_eq!(o.launch_symbol, variant.symbol());
            assert_eq!(o.exports, vec![variant.symbol()]);
            assert_eq!(o.desc["rip"]["identical_to_generator"], true, "{name}");
            assert_eq!(o.desc["rip"]["variant"], name);
        }
    }

    #[test]
    fn emu_small_run_is_bit_exact_or_an_honest_refusal() {
        for variant in ["rows", "gp"] {
            let w = new("fn:t16", &mut knobs(&[("variant", variant)])).unwrap();
            match w.emu() {
                Value::String(s) => assert!(s.starts_with("refused ("), "{s}"),
                v => {
                    assert_eq!(v["ran"], true, "{variant}: {v}");
                    assert_eq!(v["bit_exact"], true, "{variant}: {v}");
                    assert_eq!(v["reference"], REFERENCE_NAME);
                }
            }
        }
    }
}
