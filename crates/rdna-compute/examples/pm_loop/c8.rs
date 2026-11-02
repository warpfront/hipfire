// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! `iu4_c8_producer_pm` workload: the PeaceMaker twins of the exact (`cand_mode = 8`) gfx1201 IU4 C8
//! producers of the A3B F3 prefill route (`peacemaker-isa emit --kernel iu4_c8_producer_pm`, generator
//! `peacemaker_isa::kernels::iu4_c8_producer`) against the committed bundle
//! `kernels/iu4_c8_producer_pm_gfx1201.hxaco` (`rdna_compute::iu4_c8_pm::IMAGE`).
//!
//! One module, four exports; the candidate is the whole module linked by the native writer (exactly
//! `emit --co`), `--export` selects the one that is timed (all four are certified):
//!
//! | `--export` | symbol | production launcher | block | grid | kernarg |
//! |---|---|---|---|---|---|
//! | `rotate` | `iu4_c8_rotate_pm_gfx1201` | `Gpu::mq_rotate_x_i4_c8_pm` (`iu4_c8_pm.rs:435`) | 32 | `rows*K/256` | 72 B |
//! | `norm` | `iu4_c8_rmsnorm_rotate_pm_gfx1201` | `Gpu::fused_rmsnorm_mq_rotate_i4_c8_pm` (`:463`) | 256 | `rows` | 72 B |
//! | `sigmoid_rotate` | `iu4_c8_sigmoid_rotate_pm_gfx1201` | `Gpu::mq_sigmoid_rotate_x_i4_c8_pm` (`:494`) | 32 | `rows*K/256` | 80 B |
//! | `norm_out` | `iu4_c8_rmsnorm_out_rotate_pm_gfx1201` | `Gpu::fused_rmsnorm_out_mq_rotate_i4_c8_pm` (`:524`) | 256 | `rows` | 80 B |
//!
//! Grid/block/kernarg are those of `launch_iu4_c8_pm` (`iu4_c8_pm.rs:164-194` kernarg block, `:595-612`
//! args and grid/block, `:621-628` launch with dynamic LDS 0: the norm exports' LDS is the object's
//! static segment). Kernarg: `x@0 rotated@8 sidecar@16 signs1@24 signs2@32 weight@40 awq@48 rows@56
//! K@60 eps@64 pad@68`, plus one pointer `@72` for the 80 B exports (`gate` of `sigmoid_rotate`,
//! `norm_out` of `norm_out`).
//!
//! # Knobs
//! * `--export rotate|norm|sigmoid_rotate|norm_out` (default `rotate`): the export that is launched.
//! * `--ref-rows full|sample:<n>` (default `full`): the CPU reference covers every row, or rows
//!   `0..16`, the last 16 and `n` evenly spread rows (every byte of those rows' outputs: the f32
//!   rotated row, its `K/128` sidecar records, and for `norm_out` the normalized row). The bitwise
//!   baseline-identity check always covers the whole output.
//!
//! There are no `--base-*` knobs: the baseline is always the shipped bundle's export of the same name.
//!
//! # Shape grammar
//! `[a3b:NAME][,rows=<n>][,K=<k>][,awq=0|1]` (items in any order; pairs override the preset).
//! An empty shape is `a3b:chunk`. `rows` is `1..=8192` (`iu4_c8_pm.rs:64`), `K` must be one the
//! export serves: 2048 or 4096 for `rotate` / `sigmoid_rotate`, 2048 for `norm` / `norm_out`
//! (`iu4_c8_producer::validate`, `iu4_c8_pm.rs:76-83`). `awq=1` (default) passes the AWQ scale table
//! (`awq_scale` of the consumer weight, positive f32 `[K]`), `awq=0` passes null.
//!
//! The real F3 call sites fix `K` per export; `NAME` fixes `rows`:
//!
//! | preset | rows | source |
//! |---|---|---|
//! | `a3b:chunk` | 8192 | first pp16000 chunk and the A3B row ceiling: `moe_program.rs:5210` (`A3B_ROWS = 8192`), `shared_down_sigmoid_epi_oracle.rs:53-54`, `iu4_c8_pm_oracle.rs:67` |
//! | `a3b:tail` | 7808 | second pp16000 chunk (16000 - 8192): same oracle docs |
//! | `a3b:chunk4k` | 4096 | the memory-aware resident-MoE chunk: `docs/env-vars.md:471` (`HIPFIRE_PREFILL_CHUNK_ROWS`) |
//!
//! | export | default `K` | call site |
//! |---|---|---|
//! | `rotate` | 4096 | the FA `wo` input rotate `try_gfx12_rotate_exact_prepared_no_epilogue` (`batched_moe.rs:114-131`, called at `:2130-2137` with `layer.wo.k`; the FA output dim is H16 x D256 = 4096, the geometry of `peacemaker-isa.rs:167` FA qprep); `K=2048` is the shared/router input basis `rotate_x_mq_i4_gfx12_exact_batched` of `moe_program.rs:866-878` (`weight.k` = the gate/up `K`, 2048: `moe_program.rs:5253`) |
//! | `sigmoid_rotate` | 4096 | the L4 fold of the FA output gate into the `wo` rotate: `batched_moe.rs:152-183`, `:2086-2094` (`layer.wo.k`) |
//! | `norm` | 2048 | the LA qkvza / FA qkv input producer `try_gfx12_rmsnorm_exact_prepared`: `batched_moe.rs:531-542` and `:1705-1716` (`dim` = hidden 2048) |
//! | `norm_out` | 2048 | the L4 norm fold into the input basis: `moe_program.rs:744-809` (`k = weight.k`, the shared gate `K`) |
//!
//! **Flash-Next (hidden 2560) is not served by this producer**: `K` must be 2048 or 4096, so there is no
//! `fn:` preset; `fn:NAME` and `K=2560` are refused with that reason.
//!
//! # Inputs
//! Deterministic (`Xs64`): `x` f32 `[rows, K]` uniform in `[-1, 1)` scaled per row by
//! `0.25 * (1 + row % 5)` (the oracle's `random` class, `iu4_c8_pm_oracle.rs:478-488`); `signs1` /
//! `signs2` the production `+-1.0` FWHT tables, 256 f32 each, seeds 42 / 1042 (`scratch.rs:1353-1365`,
//! `:2008-2025`); the norm weight and the AWQ table positive f32 `[K]` in `[0.5, 2.0)`; the sigmoid gate
//! f32 `[rows, K]` uniform in `[-12, 12)`. `eps = 1e-6` for the norm exports (`iu4_c8_pm_oracle.rs:124`),
//! `0` otherwise (what the launcher passes for the rotate exports). Outputs start at a `0xA5` sentinel
//! (the oracle's `SENTINEL`) so a skipped store cannot match by accident.
//!
//! # Outputs and pass criterion
//! Compared: the f32 rotated row `[rows, K]`, the `block_i4_128` sidecar (`K/128 * rows` records of 72 B,
//! record `(2*group + h) * rows + row`), and for `norm_out` the normalized f32 row. All bitwise equal to
//! the shipped export's, and equal to the CPU reference below.
//!
//! # CPU reference (exact for the arithmetic; see per-export notes)
//! Transcribed from `crates/peacemaker-isa/tests/iu4_c8_producer.rs` (`reference_rotate` `:641-663`,
//! `reference_norm` `:688-753`, `fwht_ref` `:596-626`, `quantize_mode` `:527-593`; the quantizer also
//! equals `iu4_c8_pm_oracle.rs:575-657`, which that oracle checks against the incumbent's own bytes on
//! the card) and the incumbent recipe documented in `iu4_c8_producer.rs:36-66`. Pure f32 IEEE
//! arithmetic (`fma` as the generator does), so exact for `rotate`. The other exports also use the
//! emulator's integer models of the transcendentals `v_rsq_f32` (`norm`, `norm_out`) and `v_exp_f32`
//! (`sigmoid_rotate`, hipcc's range-reduced `expf` after `gdn_chunk_prep.rs:382-429`):
//! `peacemaker_ir::emu::trans` documents these as candidate models not yet qualified bit-exact against
//! the card, so a reference mismatch that coexists with baseline identity points at that model, not at
//! the candidate. `norm` uses the fused-RMSNorm reduction order (wave tree, then the eight wave sums),
//! `norm_out` the standalone `rmsnorm_f32` order (`kernels/src/rmsnorm.hip:13-21`: LDS tree
//! `s = 128 .. 1`); the repo has no CPU reference for the two folded exports other than this one
//! (`iu4_c8_pm_oracle` proves them against the HIP chain on the card).
//!
//! # `--emu`
//! The candidate export runs on the emulator at `rows = 8, K = 2048` (this run's `awq`) over every
//! workgroup and is compared with the CPU reference bit for bit; an opcode the emulator does not model
//! is reported as `refused (...)`.

use crate::common::{
    emitted_object, emu_run, i4_record, no_leftover, put_f32, put_i32, put_ptr, put_u32, read_rip,
    refused, rip_object, shipped_object, take_dim, Elem, EmuBuf, RipSource, ShapeSpec, Xs64,
};
use crate::prod::{boxed, buffers_json, Covered, Expect, OutBuf, Producer, Ptrs};
use crate::workload::{Knobs, Launch, Object, Result, Workload};
use peacemaker_ir::emu::trans;
use peacemaker_isa::kernels::iu4_c8_producer::{self as c8gen, Export};
use peacemaker_isa::Arch;
use rayon::prelude::*;
use serde_json::{json, Value};
use std::ops::Range;

const ARCH: Arch = Arch::Gfx1201;
const IR_ARCH: peacemaker_ir::Arch = peacemaker_ir::Arch::Gfx1201;
/// RMSNorm epsilon of the norm exports (`iu4_c8_pm_oracle.rs:124`).
const EPS: f32 = 1.0e-6;
/// Pre-launch fill of every output (`iu4_c8_pm_oracle.rs:123`).
const SENTINEL: u8 = 0xa5;
/// Bytes of one `block_i4_128` record.
const RECORD: usize = 72;
/// Rows at each end of the matrix that a sampled reference always covers.
const SAMPLE_EDGE: usize = 16;
/// Where production selects the shipped object (`iu4_c8_pm.rs:54` image, `:589-591` load of the symbol,
/// `:621-628` launch) and which production gate reaches each export.
const SELECTION: &str = "crates/rdna-compute/src/iu4_c8_pm.rs:54 IMAGE, :589-591 \
     ensure_embedded_kernel(MODULE, IMAGE, symbol), :621-628 launch_blob_recorded; admitted at \
     gemv.rs:7029-7032 (rotate), gemv.rs:4513-4525 (norm), batched_moe.rs:174-179 (sigmoid_rotate), \
     moe_program.rs:779-805 (norm_out)";
const CANDIDATE: &str = "peacemaker_isa::kernels::iu4_c8_producer::emit_module + native::assemble \
     (peacemaker-isa emit --kernel iu4_c8_producer_pm --co)";

/// One input buffer of the launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum In {
    X,
    Signs1,
    Signs2,
    Weight,
    Awq,
    Gate,
}

impl In {
    fn name(self) -> &'static str {
        match self {
            Self::X => "x",
            Self::Signs1 => "signs1",
            Self::Signs2 => "signs2",
            Self::Weight => "weight",
            Self::Awq => "awq",
            Self::Gate => "gate",
        }
    }
    fn bytes(self, rows: usize, k: usize) -> usize {
        match self {
            Self::X | Self::Gate => rows * k * 4,
            Self::Signs1 | Self::Signs2 => 256 * 4,
            Self::Weight | Self::Awq => k * 4,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RefRows {
    Full,
    Sample(usize),
}

impl RefRows {
    fn parse(s: &str) -> Result<Self> {
        if s == "full" {
            return Ok(Self::Full);
        }
        match s
            .strip_prefix("sample:")
            .and_then(|n| n.parse::<usize>().ok())
        {
            Some(n) if n > 0 => Ok(Self::Sample(n)),
            _ => Err(format!(
                "--ref-rows {s}: expected full or sample:<n> with n >= 1"
            )),
        }
    }
    fn label(self) -> String {
        match self {
            Self::Full => "full".into(),
            Self::Sample(n) => format!("sample:{n}"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Shape {
    rows: usize,
    k: usize,
    awq: bool,
    /// The preset the rows came from (`a3b:chunk` when none was given).
    preset: String,
}

/// `K` of the export's F3 call site (see the module docs).
fn default_k(export: Export) -> usize {
    if export.is_norm() {
        2048
    } else {
        4096
    }
}

fn parse_shape(export: Export, s: &str) -> Result<Shape> {
    let spec = ShapeSpec::parse(s)?;
    let (label, rows) = match &spec.preset {
        None => ("a3b:chunk".to_owned(), 8192),
        Some((fam, name)) if fam == "fn" => {
            return Err(format!(
                "shape `fn:{name}`: Flash-Next (hidden 2560) is not served by iu4_c8_producer_pm: \
                 its K must be 2048 or 4096 (rotate, sigmoid_rotate) or 2048 (norm, norm_out), so \
                 there is no fn: preset; use a3b:chunk|tail|chunk4k"
            ))
        }
        Some((_, name)) => {
            let rows = match name.as_str() {
                "chunk" => 8192,
                "tail" => 7808,
                "chunk4k" => 4096,
                other => {
                    return Err(format!(
                        "shape: unknown a3b preset `{other}` (chunk|tail|chunk4k)"
                    ))
                }
            };
            (format!("a3b:{name}"), rows)
        }
    };
    let mut m = spec.resolve(&[("rows", rows), ("K", default_k(export)), ("awq", 1)]);
    let (rows, k, awq) = (
        take_dim(&mut m, "rows")?,
        take_dim(&mut m, "K")?,
        take_dim(&mut m, "awq")?,
    );
    no_leftover(&m)?;
    if awq > 1 {
        return Err(format!("shape: awq={awq}: expected 0 or 1"));
    }
    c8gen::validate(export, k, rows).map_err(|e| {
        if k == 2560 {
            format!("{e}; Flash-Next (hidden 2560) is not served by iu4_c8_producer_pm")
        } else {
            e
        }
    })?;
    Ok(Shape {
        rows,
        k,
        awq: awq == 1,
        preset: label,
    })
}

#[derive(Clone, Debug)]
struct C8 {
    export: Export,
    shape: Shape,
    ref_rows: RefRows,
    /// `--rip`: the candidate is this script's export, not the Rust emit.
    rip: RipSource,
}

/// `--kernel iu4_c8_producer_pm` (`rip` = `--rip FILE.rip`: the export's twin).
#[cfg(test)]
pub fn new(shape: &str, knobs: &mut Knobs) -> Result<Box<dyn Workload>> {
    new_with_rip(shape, knobs, None)
}

pub fn new_with_rip(
    shape: &str,
    knobs: &mut Knobs,
    rip: Option<&std::path::Path>,
) -> Result<Box<dyn Workload>> {
    let mut c8 = C8::build(shape, knobs)?;
    c8.rip = read_rip(rip)?;
    Ok(boxed(c8))
}

impl C8 {
    fn build(shape: &str, knobs: &mut Knobs) -> Result<Self> {
        let tag = knobs.take_or("export", "rotate");
        let export = c8gen::module_exports()
            .into_iter()
            .find(|e| e.tag() == tag)
            .ok_or_else(|| {
                format!("--export {tag}: expected rotate|norm|sigmoid_rotate|norm_out")
            })?;
        let ref_rows = RefRows::parse(&knobs.take_or("ref-rows", "full"))?;
        Ok(Self {
            export,
            shape: parse_shape(export, shape)?,
            ref_rows,
            rip: None,
        })
    }

    /// Input buffers in the order of [`Producer::inputs`].
    fn layout(&self) -> Vec<In> {
        let mut v = vec![In::X, In::Signs1, In::Signs2];
        if self.export.is_norm() {
            v.push(In::Weight);
        }
        if self.shape.awq {
            v.push(In::Awq);
        }
        if self.export == Export::SigmoidRotate {
            v.push(In::Gate);
        }
        v
    }

    /// `eps` as the launcher passes it: the norm's, else `0.0` (`iu4_c8_pm.rs:453`, `:513`).
    fn eps(&self) -> f32 {
        if self.export.is_norm() {
            EPS
        } else {
            0.0
        }
    }

    /// `(grid, block)` of `launch_iu4_c8_pm` (`iu4_c8_pm.rs:608-612`).
    fn geometry(&self) -> ([u32; 3], [u32; 3]) {
        let Shape { rows, k, .. } = self.shape;
        let block = [self.export.block(), 1, 1];
        if self.export.is_norm() {
            ([rows as u32, 1, 1], block)
        } else {
            ([(k / 256 * rows) as u32, 1, 1], block)
        }
    }

    fn seed(&self, salt: u64) -> u64 {
        0xc8_0000_0000 + ((self.shape.k as u64) << 20) + ((self.shape.rows as u64) << 4) + salt
    }

    /// The run at the emulator's small shape: `rows = 8, K = 2048`, this run's AWQ, full reference.
    fn small(&self) -> Self {
        Self {
            export: self.export,
            shape: Shape {
                rows: 8,
                k: 2048,
                awq: self.shape.awq,
                preset: "emu".into(),
            },
            ref_rows: RefRows::Full,
            rip: self.rip.clone(),
        }
    }

    fn emulate(&self) -> Result<Value> {
        let small = self.small();
        let obj = small.candidate()?;
        let (inputs, outs, layout) = (small.inputs(), small.outputs(), small.layout());
        let in_va = |i: usize| 0x2000_0000u64 + i as u64 * 0x100_0000;
        let out_va = |i: usize| 0x4000_0000u64 + i as u64 * 0x100_0000;
        let launch = small.launch(&Ptrs {
            inputs: (0..inputs.len()).map(in_va).collect(),
            outputs: (0..outs.len()).map(out_va).collect(),
        });
        let mut bufs: Vec<EmuBuf> = layout
            .iter()
            .zip(&inputs)
            .enumerate()
            .map(|(i, (w, bytes))| EmuBuf {
                name: w.name(),
                va: in_va(i),
                bytes: bytes.clone(),
            })
            .collect();
        bufs.extend(outs.iter().enumerate().map(|(i, o)| EmuBuf {
            name: o.name,
            va: out_va(i),
            bytes: o.init.clone(),
        }));
        let mem = emu_run(
            &obj.bytes,
            &obj.launch_symbol,
            launch.kernarg,
            bufs,
            launch.grid,
            launch.block,
            launch.lds,
        )?;
        let expect = small.reference(&inputs, &outs)?;
        let mut mismatching = 0;
        for (i, o) in outs.iter().enumerate() {
            let want = expect.outputs[i]
                .as_ref()
                .ok_or_else(|| format!("the reference does not cover {}", o.name))?;
            mismatching += mismatches(o.elem, mem.read(out_va(i), o.init.len())?, want);
        }
        Ok(json!({
            "ran": true,
            "bit_exact": mismatching == 0,
            "shape": {"rows": 8, "K": 2048, "awq": small.shape.awq},
            "symbol": obj.launch_symbol,
            "reference": expect.reference,
            "mismatching_elements": mismatching,
        }))
    }
}

/// Elements of `got` (of type `elem`) that differ from `want` on its covered ranges.
fn mismatches(elem: Elem, got: &[u8], want: &Covered) -> usize {
    let w = elem.width();
    want.ranges
        .iter()
        .map(|r| {
            got[r.clone()]
                .chunks(w)
                .zip(want.bytes[r.clone()].chunks(w))
                .filter(|(a, b)| a != b)
                .count()
        })
        .sum()
}

fn sentinel(name: &'static str, bytes: usize, elem: Elem) -> OutBuf {
    OutBuf {
        name,
        init: vec![SENTINEL; bytes],
        elem,
        compare: true,
    }
}

impl Producer for C8 {
    fn kernel(&self) -> &'static str {
        c8gen::KERNEL_ID
    }

    fn shape_json(&self) -> Value {
        let Shape { rows, k, awq, .. } = &self.shape;
        let inputs: Vec<(&str, usize)> = self
            .layout()
            .into_iter()
            .map(|w| (w.name(), w.bytes(*rows, *k)))
            .collect();
        let (grid, block) = self.geometry();
        json!({
            "rows": rows,
            "K": k,
            "awq": awq,
            "preset": self.shape.preset,
            "eps": self.eps(),
            "grid": grid,
            "block": block,
            "kernarg_bytes": self.export.kernarg_bytes(),
            "buffers": buffers_json(&inputs, &self.outputs()),
        })
    }

    fn knobs_json(&self) -> Value {
        json!({
            "export": self.export.tag(),
            "symbol": self.export.symbol(),
            "ref_rows": self.ref_rows.label(),
        })
    }

    fn candidate(&self) -> Result<Object> {
        if let Some(rip) = &self.rip {
            return rip_object(
                rip,
                ARCH,
                self.export.tag(),
                self.export.symbol(),
                &c8gen::emit(ARCH, self.export)?.s_text,
                self.knobs_json(),
            );
        }
        let (_, text, _) = c8gen::emit_module(ARCH)?;
        let exports: Vec<String> = c8gen::module_exports()
            .iter()
            .map(|e| e.symbol().to_owned())
            .collect();
        emitted_object(
            &text,
            ARCH,
            self.export.symbol(),
            &exports,
            CANDIDATE,
            self.knobs_json(),
        )
    }

    fn baseline(&self) -> Result<Object> {
        let exports: Vec<&str> = c8gen::module_exports().iter().map(|e| e.symbol()).collect();
        shipped_object(
            rdna_compute::iu4_c8_pm::IMAGE,
            "kernels/iu4_c8_producer_pm_gfx1201.hxaco",
            SELECTION,
            self.export.symbol(),
            &exports,
            json!({"export": self.export.tag()}),
        )
    }

    fn inputs(&self) -> Vec<Vec<u8>> {
        let Shape { rows, k, .. } = self.shape;
        self.layout()
            .into_iter()
            .map(|w| match w {
                In::X => activations(rows, k, self.seed(1)),
                In::Signs1 => f32_bytes(&fwht_signs(42)),
                In::Signs2 => f32_bytes(&fwht_signs(1042)),
                In::Weight => positive_table(self.seed(2), k),
                In::Awq => positive_table(self.seed(3), k),
                In::Gate => gate_rows(rows, k, self.seed(4)),
            })
            .collect()
    }

    fn outputs(&self) -> Vec<OutBuf> {
        let Shape { rows, k, .. } = self.shape;
        let mut v = vec![
            sentinel("rotated", rows * k * 4, Elem::F32),
            sentinel("sidecar", k / 128 * rows * RECORD, Elem::U8),
        ];
        if self.export == Export::NormOut {
            v.push(sentinel("norm_out", rows * k * 4, Elem::F32));
        }
        v
    }

    fn launch(&self, p: &Ptrs) -> Launch {
        let layout = self.layout();
        let input = |w: In| {
            layout
                .iter()
                .position(|x| *x == w)
                .map_or(0, |i| p.inputs[i])
        };
        let mut k = Vec::with_capacity(80);
        for ptr in [
            input(In::X),
            p.outputs[0],
            p.outputs[1],
            input(In::Signs1),
            input(In::Signs2),
            input(In::Weight),
            input(In::Awq),
        ] {
            put_ptr(&mut k, ptr);
        }
        put_i32(&mut k, self.shape.rows as i32);
        put_i32(&mut k, self.shape.k as i32);
        put_f32(&mut k, self.eps());
        put_u32(&mut k, 0);
        match self.export {
            Export::SigmoidRotate => put_ptr(&mut k, input(In::Gate)),
            Export::NormOut => put_ptr(&mut k, p.outputs[2]),
            Export::Rotate | Export::Norm => {}
        }
        let (grid, block) = self.geometry();
        Launch {
            kernarg: k,
            grid,
            block,
            lds: 0,
        }
    }

    fn reference(&self, inputs: &[Vec<u8>], outputs: &[OutBuf]) -> Result<Expect> {
        let Shape { rows, k, .. } = self.shape;
        let layout = self.layout();
        if inputs.len() != layout.len() {
            return Err(format!(
                "reference: {} inputs, expected {}",
                inputs.len(),
                layout.len()
            ));
        }
        let input = |w: In| {
            layout
                .iter()
                .position(|x| *x == w)
                .map(|i| inputs[i].as_slice())
        };
        let table = |w: In| input(w).map(f32s);
        let model = Model {
            export: self.export,
            k,
            eps: self.eps(),
            x: input(In::X).ok_or("reference: no x")?,
            gate: input(In::Gate),
            s1: table(In::Signs1).ok_or("reference: no signs1")?,
            s2: table(In::Signs2).ok_or("reference: no signs2")?,
            weight: table(In::Weight),
            awq: table(In::Awq),
        };
        let picked: Vec<usize> = match self.ref_rows {
            RefRows::Full => (0..rows).collect(),
            RefRows::Sample(n) => sample_rows(rows, n),
        };
        let done: Vec<RowOut> = picked.par_iter().map(|&r| model.row(r)).collect();

        let mut rot = vec![0u8; rows * k * 4];
        let mut sidecar = vec![0u8; k / 128 * rows * RECORD];
        let mut norm = (self.export == Export::NormOut).then(|| vec![0u8; rows * k * 4]);
        for (&r, o) in picked.iter().zip(&done) {
            rot[r * k * 4..(r + 1) * k * 4].copy_from_slice(&o.rot);
            for (kb, rec) in o.recs.chunks_exact(RECORD).enumerate() {
                let at = (kb * rows + r) * RECORD;
                sidecar[at..at + RECORD].copy_from_slice(rec);
            }
            if let (Some(dst), Some(src)) = (norm.as_mut(), &o.norm) {
                dst[r * k * 4..(r + 1) * k * 4].copy_from_slice(src);
            }
        }
        let want: Vec<usize> = std::iter::once(rot.len())
            .chain(std::iter::once(sidecar.len()))
            .chain(norm.as_ref().map(Vec::len))
            .collect();
        let have: Vec<usize> = outputs.iter().map(|o| o.init.len()).collect();
        if want != have {
            return Err(format!(
                "reference: buffers {want:?} bytes, outputs {have:?}"
            ));
        }

        let (row_ranges, record_ranges): (Vec<Range<usize>>, Vec<Range<usize>>) =
            if self.ref_rows == RefRows::Full {
                (vec![0..rot.len()], vec![0..sidecar.len()])
            } else {
                (
                    picked.iter().map(|&r| r * k * 4..(r + 1) * k * 4).collect(),
                    picked
                        .iter()
                        .flat_map(|&r| {
                            (0..k / 128).map(move |kb| {
                                let at = (kb * rows + r) * RECORD;
                                at..at + RECORD
                            })
                        })
                        .collect(),
                )
            };
        let cover = |bytes: Vec<u8>, ranges: &[Range<usize>]| Covered {
            bytes,
            ranges: ranges.to_vec(),
        };
        let mut out = vec![
            Some(cover(rot, &row_ranges)),
            Some(cover(sidecar, &record_ranges)),
        ];
        if let Some(n) = norm {
            out.push(Some(cover(n, &row_ranges)));
        }
        Ok(Expect {
            outputs: out,
            reference: match self.export {
                Export::Rotate => "cpu_c8_rotate_fwht256",
                Export::Norm => "cpu_c8_rmsnorm_rotate(v_rsq=ir_trans_model)",
                Export::SigmoidRotate => "cpu_c8_sigmoid_rotate(v_exp=ir_trans_model)",
                Export::NormOut => "cpu_c8_rmsnorm_out_rotate(v_rsq=ir_trans_model)",
            }
            .into(),
        })
    }

    fn emu(&self) -> Value {
        match self.emulate() {
            Ok(v) => v,
            Err(reason) => refused(reason.split(" | ").next().unwrap_or(&reason)),
        }
    }
}

// ---------------------------------------------------------------------------
// Inputs

/// The production FWHT sign tables: 256 `+-1.0` (`scratch.rs:1353-1365` `gen_fwht_signs`, seeds 42 and
/// 1042 at `:2018-2019`).
fn fwht_signs(seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..256)
        .map(|_| {
            state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fffffff;
            if (state >> 16) & 1 == 1 {
                1.0
            } else {
                -1.0
            }
        })
        .collect()
}

/// `[rows, k]` f32, uniform in `[-1, 1)` scaled per row by `0.25 * (1 + row % 5)`.
fn activations(rows: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut rng = Xs64::new(seed);
    let mut out = Vec::with_capacity(rows * k * 4);
    for row in 0..rows {
        let scale = 0.25 * (1 + row % 5) as f32;
        for _ in 0..k {
            out.extend_from_slice(&(rng.unit() * scale).to_le_bytes());
        }
    }
    out
}

/// `[rows, k]` f32 gate logits, uniform in `[-12, 12)`.
fn gate_rows(rows: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut rng = Xs64::new(seed);
    (0..rows * k)
        .flat_map(|_| (rng.unit() * 12.0).to_le_bytes())
        .collect()
}

/// `[k]` f32, positive, uniform in `[0.5, 2.0)` (norm weight, AWQ scales).
fn positive_table(seed: u64, k: usize) -> Vec<u8> {
    let mut rng = Xs64::new(seed);
    (0..k)
        .flat_map(|_| (0.5 + (rng.unit() + 1.0) * 0.75).to_le_bytes())
        .collect()
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn row_f32(bytes: &[u8], row: usize, k: usize) -> Vec<f32> {
    f32s(&bytes[row * k * 4..(row + 1) * k * 4])
}

/// Rows `0..16`, the last 16 and `n` evenly spread rows, sorted and unique.
fn sample_rows(rows: usize, n: usize) -> Vec<usize> {
    let mut v: Vec<usize> = (0..SAMPLE_EDGE.min(rows))
        .chain(rows.saturating_sub(SAMPLE_EDGE)..rows)
        .chain((0..n).map(|i| i * rows / n))
        .collect();
    v.sort_unstable();
    v.dedup();
    v
}

// ---------------------------------------------------------------------------
// CPU reference

/// The inputs of one export's reference (`k`-element tables parsed once, `x` / `gate` read per row).
struct Model<'a> {
    export: Export,
    k: usize,
    eps: f32,
    x: &'a [u8],
    gate: Option<&'a [u8]>,
    s1: Vec<f32>,
    s2: Vec<f32>,
    weight: Option<Vec<f32>>,
    awq: Option<Vec<f32>>,
}

/// One row's expected bytes: the rotated f32 row, its `k / 128` records (consecutive, block order) and,
/// for `norm_out`, the normalized row.
struct RowOut {
    rot: Vec<u8>,
    recs: Vec<u8>,
    norm: Option<Vec<u8>>,
}

impl Model<'_> {
    fn row(&self, row: usize) -> RowOut {
        let k = self.k;
        let x = row_f32(self.x, row, k);
        let (y, norm) = match self.export {
            Export::Rotate => (x, None),
            Export::SigmoidRotate => {
                let gate = self.gate.expect("sigmoid_rotate has a gate");
                let g = row_f32(gate, row, k);
                (
                    x.iter().zip(&g).map(|(x, g)| x * sigmoid(*g)).collect(),
                    None,
                )
            }
            Export::Norm | Export::NormOut => {
                let w = self
                    .weight
                    .as_ref()
                    .expect("the norm exports have a weight");
                let rms = rms_scale(&x, self.export, self.eps);
                let y: Vec<f32> = x.iter().zip(w).map(|(x, w)| x * w * rms).collect();
                let norm = (self.export == Export::NormOut).then(|| f32_bytes(&y));
                (y, norm)
            }
        };
        let mut rot = Vec::with_capacity(k * 4);
        let mut recs = Vec::with_capacity(k / 128 * RECORD);
        for g in 0..k / 256 {
            let mut v = [[0f32; 8]; 32];
            for (p, lane) in v.iter_mut().enumerate() {
                for (e, slot) in lane.iter_mut().enumerate() {
                    let base = g * 256 + 8 * p + e;
                    let t = match &self.awq {
                        Some(a) => y[base] / a[base],
                        None => y[base],
                    };
                    *slot = t * self.s1[8 * p + e];
                }
            }
            let f = fwht(&mut v, &self.s2);
            for h in 0..2 {
                let block: &[f32; 128] = f[128 * h..128 * (h + 1)].try_into().unwrap();
                recs.extend_from_slice(&quantize_c8(block));
            }
            rot.extend(f.iter().flat_map(|x| x.to_le_bytes()));
        }
        RowOut { rot, recs, norm }
    }
}

/// The FWHT body shared by the incumbent kernels (`tests/iu4_c8_producer.rs:596-626`): lane `p` holds
/// `v[p][0..8]`; in-lane butterflies strides 1, 2, 4, lane butterflies at XOR 1 .. 16, then
/// `(v * 0.0625) * signs2`. Returns `f[8p + i]`.
fn fwht(v: &mut [[f32; 8]; 32], signs2: &[f32]) -> [f32; 256] {
    for lane in v.iter_mut() {
        for stride in [1usize, 2, 4] {
            for lo in (0..8).filter(|i| i & stride == 0) {
                let (a, b) = (lane[lo], lane[lo + stride]);
                lane[lo] = a + b;
                lane[lo + stride] = a - b;
            }
        }
    }
    for stride in [1usize, 2, 4, 8, 16] {
        let prev = *v;
        for p in 0..32 {
            for i in 0..8 {
                let partner = prev[p ^ stride][i];
                v[p][i] = if p & stride != 0 {
                    partner - prev[p][i]
                } else {
                    prev[p][i] + partner
                };
            }
        }
    }
    let mut f = [0f32; 256];
    for p in 0..32 {
        for i in 0..8 {
            f[8 * p + i] = v[p][i] * 0.0625f32 * signs2[8 * p + i];
        }
    }
    f
}

/// `q = clamp(rint(x / d), -8, 7)` as float (`tests/iu4_c8_producer.rs:504-506`, `:530`).
fn c8_code(x: f32, d: f32) -> f32 {
    (x / d).round_ties_even().clamp(-8.0, 7.0)
}

/// One quantizer lane's four-element mse: `err = fma(-q, dj, x)`, `mse = fma(err, err, mse)` from `+0`.
fn lane_mse(xs: &[f32], dj: f32) -> f32 {
    xs.iter().fold(0.0f32, |mse, &x| {
        let err = (-c8_code(x, dj)).mul_add(dj, x);
        err.mul_add(err, mse)
    })
}

/// Wave32 XOR-butterfly sum, offsets 16, 8, 4, 2, 1 (every lane ends equal; lane 0 is returned).
fn xor_sum32(mut v: [f32; 32]) -> f32 {
    for off in [16usize, 8, 4, 2, 1] {
        let prev = v;
        for (l, x) in v.iter_mut().enumerate() {
            *x = prev[l] + prev[l ^ off];
        }
    }
    v[0]
}

/// `quantize_block_i4_128_wave<false>` (the C8 search) of 128 rotated floats into one 72 B record
/// (`tests/iu4_c8_producer.rs:527-593`; `iu4_c8_pm_oracle.rs:575-657`). `amax == 0` forces zero codes and
/// `d = 1`; otherwise the codes are recomputed from `best_d` (which stays `1.0` if no candidate beats the
/// `1e30` sentinel).
fn quantize_c8(x: &[f32; 128]) -> [u8; RECORD] {
    let amax = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let mut best_d = 1.0f32;
    let mut q = [0i32; 128];
    if amax != 0.0 {
        let mut best_mse = 1.0e30f32;
        for j in 0..8u32 {
            let dj = (amax / 7.0f32) * 0.5f32 * (1.0f32 + (j as f32) / 7.0f32);
            let mut lanes = [0f32; 32];
            for (l, m) in lanes.iter_mut().enumerate() {
                *m = lane_mse(&x[4 * l..4 * l + 4], dj);
            }
            let mse = xor_sum32(lanes);
            if mse < best_mse {
                best_mse = mse;
                best_d = dj;
            }
        }
        for (qi, v) in q.iter_mut().zip(x) {
            *qi = c8_code(*v, best_d) as i32;
        }
    }
    i4_record(best_d, &q)
}

/// `shfl_down` of a wave32 by `off` (an out-of-range source keeps the lane's own value).
fn shfl_down(v: &[f32; 32], off: usize) -> [f32; 32] {
    std::array::from_fn(|l| if l + off < 32 { v[l + off] } else { v[l] })
}

/// Lane 0 of `v += shfl_down(v, off)` over `offs`.
fn tree_down(mut v: [f32; 32], offs: &[usize]) -> f32 {
    for &off in offs {
        let down = shfl_down(&v, off);
        for (x, d) in v.iter_mut().zip(down) {
            *x += d;
        }
    }
    v[0]
}

/// The row's sum of squares from the 256 thread partials, in the reduction order of the export:
/// `norm` is the fused RMSNorm's (each wave's `shfl_down` tree, then the eight wave sums by
/// `shfl_down` 4, 2, 1: `tests/iu4_c8_producer.rs:704-721`); `norm_out` the standalone `rmsnorm_f32`'s
/// (`rmsnorm.hip:15-21`: `sdata[t] += sdata[t + s]`, `s = 128 .. 1`).
fn reduce_sum(partial: &[f32; 256], export: Export) -> f32 {
    if export == Export::Norm {
        let waves: [f32; 8] = std::array::from_fn(|w| {
            tree_down(
                std::array::from_fn(|l| partial[32 * w + l]),
                &[16, 8, 4, 2, 1],
            )
        });
        let t: [f32; 32] = std::array::from_fn(|l| if l < 8 { waves[l] } else { 0.0 });
        tree_down(t, &[4, 2, 1])
    } else {
        let mut sd = *partial;
        let mut s = 128;
        while s > 0 {
            for t in 0..s {
                sd[t] += sd[t + s];
            }
            s >>= 1;
        }
        sd[0]
    }
}

/// `rms = rsqrt(sum / n + eps)` of one row: per-thread `fma` chain over `i = tid + 256 j`
/// (`tests/iu4_c8_producer.rs:695-703`), [`reduce_sum`], then the incumbent's vector `v_rsq_f32` with its
/// denormal guard (`:722-735`).
fn rms_scale(x: &[f32], export: Export, eps: f32) -> f32 {
    let partial: [f32; 256] = std::array::from_fn(|tid| {
        (tid..x.len())
            .step_by(256)
            .fold(0.0f32, |s, i| x[i].mul_add(x[i], s))
    });
    let mean = reduce_sum(&partial, export) / (x.len() as f32) + eps;
    let small = mean < f32::from_bits(0x0080_0000);
    let scaled = if small {
        mean * f32::from_bits(0x4b80_0000)
    } else {
        mean
    };
    let rms = f32::from_bits(trans::rsq_f32(IR_ARCH, scaled.to_bits()));
    if small {
        rms * f32::from_bits(0x4580_0000)
    } else {
        rms
    }
}

/// `x * 2^n` rounded once (`tests/gdn_chunk_prep.rs:382-384`).
fn ldexp(x: f32, n: i32) -> f32 {
    (f64::from(x) * 2f64.powi(n.clamp(-400, 400))) as f32
}

/// hipcc's `expf(x)`, op for op as the generator emits it (`iu4_c8_producer.rs:554-619`; the same graph as
/// `tests/gdn_chunk_prep.rs:400-429` with `neg = false`), `v_exp_f32` through the emulator's model. The
/// range selects are written as the negated compares `v_cmp_ngt` / `v_cmp_nlt` (true on NaN).
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn expf(x: f32) -> f32 {
    let (c1, c2) = (f32::from_bits(0x3fb8_aa3b), f32::from_bits(0x32a5_705f));
    let prod = x * c1;
    let lo = c1.mul_add(x, -prod);
    let n = prod.round_ties_even();
    let lo = c2.mul_add(x, lo);
    let r = (prod - n) + lo;
    let e = ldexp(
        f32::from_bits(trans::exp_f32(IR_ARCH, r.to_bits())),
        n as i32,
    );
    // The two selects of the incumbent graph.
    let e = if !(f32::from_bits(0xc2ce_8ed0) > x) {
        e
    } else {
        0.0
    };
    if !(f32::from_bits(0x42b1_7218) < x) {
        e
    } else {
        f32::INFINITY
    }
}

/// `1 / (1 + expf(-g))`, the `sigmoid_mul_f32` factor (`iu4_c8_producer.rs:935-957`: the sign flip, `expf`,
/// `+ 1.0`, the IEEE divide).
fn sigmoid(g: f32) -> f32 {
    1.0 / (1.0 + expf(-g))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::sha256_hex;

    fn knobs(kv: &[(&str, &str)]) -> Knobs {
        Knobs(
            kv.iter()
                .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
                .collect(),
        )
    }

    fn c8(export: &str, shape: &str) -> C8 {
        C8::build(shape, &mut knobs(&[("export", export)])).unwrap()
    }

    const EXPORTS: [(&str, &str); 4] = [
        ("rotate", "iu4_c8_rotate_pm_gfx1201"),
        ("norm", "iu4_c8_rmsnorm_rotate_pm_gfx1201"),
        ("sigmoid_rotate", "iu4_c8_sigmoid_rotate_pm_gfx1201"),
        ("norm_out", "iu4_c8_rmsnorm_out_rotate_pm_gfx1201"),
    ];

    #[test]
    fn shape_grammar_and_presets() {
        let dims = |export: &str, shape: &str| {
            let s = c8(export, shape).shape;
            (s.rows, s.k, s.awq, s.preset)
        };
        assert_eq!(dims("rotate", ""), (8192, 4096, true, "a3b:chunk".into()));
        assert_eq!(
            dims("sigmoid_rotate", "a3b:tail"),
            (7808, 4096, true, "a3b:tail".into())
        );
        assert_eq!(
            dims("norm", "a3b:chunk"),
            (8192, 2048, true, "a3b:chunk".into())
        );
        assert_eq!(
            dims("norm_out", "a3b:chunk4k"),
            (4096, 2048, true, "a3b:chunk4k".into())
        );
        assert_eq!(
            dims("rotate", "a3b:tail,K=2048,awq=0"),
            (7808, 2048, false, "a3b:tail".into())
        );
        assert_eq!(
            dims("rotate", "awq=0,rows=3"),
            (3, 4096, false, "a3b:chunk".into())
        );
        let err = |export: &str, shape: &str| {
            C8::build(shape, &mut knobs(&[("export", export)])).unwrap_err()
        };
        assert!(err("rotate", "fn:t4096").contains("Flash-Next"));
        assert!(err("rotate", "K=2560").contains("2560"));
        assert!(err("norm", "K=4096").contains("does not serve K 4096"));
        assert!(err("rotate", "rows=0").contains("rows 0"));
        assert!(err("rotate", "rows=8193").contains("8193"));
        assert!(err("rotate", "a3b:nope").contains("nope"));
        assert!(err("rotate", "bogus=1").contains("bogus"));
        assert!(err("rotate", "awq=2").contains("awq=2"));
    }

    #[test]
    fn knobs_select_the_symbol_of_both_objects() {
        let all: Vec<String> = EXPORTS.iter().map(|(_, s)| (*s).to_owned()).collect();
        for (tag, symbol) in EXPORTS {
            let w = new("rows=8", &mut knobs(&[("export", tag)])).unwrap();
            let (cand, base) = (w.candidate().unwrap(), w.baseline().unwrap());
            assert_eq!(cand.launch_symbol, symbol, "{tag}");
            assert_eq!(base.launch_symbol, symbol, "{tag}");
            assert_eq!(cand.exports, all, "the candidate is the whole module");
            assert_eq!(
                base.exports, all,
                "the baseline is the whole shipped bundle"
            );
            assert_eq!(w.candidate_knobs()["export"], tag);
        }
        assert_eq!(c8("rotate", "").export, Export::Rotate);
        assert!(C8::build("", &mut knobs(&[("export", "gate")])).is_err());
        assert_eq!(RefRows::parse("sample:5").unwrap(), RefRows::Sample(5));
        for bad in ["sample:0", "sample:x", "some"] {
            assert!(RefRows::parse(bad).is_err(), "{bad}");
        }
        // Unconsumed flags stay for the driver's `finish`.
        let mut rest = knobs(&[("export", "norm"), ("ref-rows", "full"), ("zzz", "1")]);
        C8::build("", &mut rest).unwrap();
        assert_eq!(rest.0.keys().collect::<Vec<_>>(), ["zzz"]);
    }

    #[test]
    fn candidate_is_a_fresh_emit_and_assemble() {
        let (_, text, _) = c8gen::emit_module(ARCH).unwrap();
        let elf = peacemaker_isa::native::assemble(&text, ARCH).unwrap();
        let cand = c8("norm_out", "").candidate().unwrap();
        assert_eq!(cand.bytes, elf);
        assert_eq!(cand.source["sha256"], sha256_hex(&elf));
        assert_eq!(cand.desc["knobs"]["export"], "norm_out");
    }

    #[test]
    fn baseline_is_the_shipped_bundle() {
        let base = c8("sigmoid_rotate", "").baseline().unwrap();
        let image = rdna_compute::iu4_c8_pm::IMAGE;
        assert_eq!(base.bytes, image);
        assert_eq!(base.source["sha256"], sha256_hex(image));
        assert_eq!(
            base.source["artifact"],
            "kernels/iu4_c8_producer_pm_gfx1201.hxaco"
        );
        assert!(base.source["source"].as_str().unwrap().contains(":589-591"));
        for (_, symbol) in EXPORTS {
            assert!(
                crate::common::contains(&base.bytes, symbol.as_bytes()),
                "{symbol}"
            );
        }
    }

    fn fake_ptrs(c: &C8) -> Ptrs {
        Ptrs {
            inputs: (1..=c.layout().len() as u64)
                .map(|i| 0x1_0000_0000 * i)
                .collect(),
            // The three outputs of the widest export (rotated, sidecar, norm_out).
            outputs: (1..=3).map(|i| 0x9_0000_0000 * i).collect(),
        }
    }

    fn u64_at(b: &[u8], at: usize) -> u64 {
        u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
    }
    fn u32_at(b: &[u8], at: usize) -> u32 {
        u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
    }

    #[test]
    fn kernarg_layout_and_grid() {
        // rotate, K 4096: x s1 s2 awq in, rotated sidecar out; weight null.
        let c = c8("rotate", "");
        let p = fake_ptrs(&c);
        let l = c.launch(&p);
        assert_eq!(l.kernarg.len(), 72);
        let ptrs: Vec<u64> = (0..7).map(|i| u64_at(&l.kernarg, 8 * i)).collect();
        assert_eq!(
            ptrs,
            [
                p.inputs[0],
                p.outputs[0],
                p.outputs[1],
                p.inputs[1],
                p.inputs[2],
                0,
                p.inputs[3]
            ]
        );
        assert_eq!(
            (
                u32_at(&l.kernarg, 56),
                u32_at(&l.kernarg, 60),
                u32_at(&l.kernarg, 64),
                u32_at(&l.kernarg, 68)
            ),
            (8192, 4096, 0f32.to_bits(), 0)
        );
        assert_eq!((l.grid, l.block, l.lds), ([8192 * 16, 1, 1], [32, 1, 1], 0));

        // No AWQ: the table pointer is null.
        let c = c8("rotate", "awq=0,rows=3,K=2048");
        let l = c.launch(&fake_ptrs(&c));
        assert_eq!(u64_at(&l.kernarg, 48), 0);
        assert_eq!(l.grid, [3 * 8, 1, 1]);

        // norm: weight @40 then awq @48, eps, block 256, grid rows.
        let c = c8("norm", "");
        let p = fake_ptrs(&c);
        let l = c.launch(&p);
        assert_eq!(l.kernarg.len(), 72);
        assert_eq!(
            (u64_at(&l.kernarg, 40), u64_at(&l.kernarg, 48)),
            (p.inputs[3], p.inputs[4])
        );
        assert_eq!(u32_at(&l.kernarg, 64), EPS.to_bits());
        assert_eq!((l.grid, l.block, l.lds), ([8192, 1, 1], [256, 1, 1], 0));

        // sigmoid_rotate: the gate rides @72 (80 B), no weight.
        let c = c8("sigmoid_rotate", "a3b:tail");
        let p = fake_ptrs(&c);
        let l = c.launch(&p);
        assert_eq!(l.kernarg.len(), 80);
        assert_eq!(
            (u64_at(&l.kernarg, 40), u64_at(&l.kernarg, 48)),
            (0, p.inputs[3])
        );
        assert_eq!(u64_at(&l.kernarg, 72), p.inputs[4]);
        assert_eq!(u32_at(&l.kernarg, 56), 7808);
        assert_eq!(l.grid, [7808 * 16, 1, 1]);

        // norm_out: the normalized row is the third output, @72.
        let c = c8("norm_out", "a3b:chunk4k");
        let p = fake_ptrs(&c);
        let l = c.launch(&p);
        assert_eq!(l.kernarg.len(), 80);
        assert_eq!(u64_at(&l.kernarg, 72), p.outputs[2]);
        assert_eq!(c.outputs().len(), 3);
        assert_eq!(l.grid, [4096, 1, 1]);

        // Sizes agree with the buffers (`K/128` records per row), and every output starts at the sentinel.
        let c = c8("rotate", "rows=8192,K=4096");
        let (rot, sidecar) = (8192 * 4096 * 4, 32 * 8192 * 72);
        assert_eq!(c.shape_json()["buffers"]["outputs"][0]["bytes"], rot);
        assert_eq!(c.shape_json()["buffers"]["outputs"][1]["bytes"], sidecar);
        let small = c8("rotate", "rows=8,K=2048");
        assert!(small
            .outputs()
            .iter()
            .all(|o| o.init.iter().all(|&b| b == SENTINEL)));
    }

    #[test]
    fn inputs_are_deterministic_and_sized() {
        let c = c8("sigmoid_rotate", "rows=4,K=2048");
        let a = c.inputs();
        assert_eq!(a, c.inputs());
        let sizes: Vec<usize> = a.iter().map(Vec::len).collect();
        assert_eq!(sizes, [4 * 2048 * 4, 1024, 1024, 2048 * 4, 4 * 2048 * 4]);
        let signs = f32s(&a[1]);
        assert!(
            signs.iter().all(|s| s.abs() == 1.0) && signs.contains(&1.0) && signs.contains(&-1.0)
        );
        assert_ne!(a[1], a[2], "signs1 and signs2 differ (seeds 42 / 1042)");
    }

    /// `x = 1` over one 256 group, all signs `+1`: the Walsh-Hadamard transform is `[256, 0, ...]`, so
    /// `f[0] = 16`; the first record then holds `d = best_d`, `s = 7`, code 7 first.
    #[test]
    fn reference_hand_checked_fwht_and_record() {
        let x = f32_bytes(&[1.0; 256]);
        let m = Model {
            export: Export::Rotate,
            k: 256,
            eps: 0.0,
            x: &x,
            gate: None,
            s1: vec![1.0; 256],
            s2: vec![1.0; 256],
            weight: None,
            awq: None,
        };
        let o = m.row(0);
        let rot = f32s(&o.rot);
        assert_eq!(rot[0], 16.0);
        assert!(rot[1..].iter().all(|v| *v == 0.0));
        assert_eq!(o.recs.len(), 2 * RECORD);
        // amax = 16: candidate 7 is d = (16/7) * 0.5 * 2 and rint(16 / d) = 7 with a vanishing error, every
        // other candidate clips and loses.
        let (r0, r1) = o.recs.split_at(RECORD);
        assert_eq!(
            f32::from_le_bytes(r0[0..4].try_into().unwrap()),
            (16.0f32 / 7.0) * 0.5 * 2.0
        );
        assert_eq!(i32::from_le_bytes(r0[4..8].try_into().unwrap()), 7);
        assert_eq!(r0[8], 7);
        assert!(r0[9..].iter().all(|&b| b == 0));
        // The all-zero block never searches: d = 1, no codes.
        assert_eq!(f32::from_le_bytes(r1[0..4].try_into().unwrap()), 1.0);
        assert!(r1[4..].iter().all(|&b| b == 0));
    }

    #[test]
    fn awq_divides_the_row_before_the_rotate() {
        // x = 2, awq = 2, signs1 = -1: the rotate input is -1 (FWHT of -1s: f[0] = -16).
        let x = f32_bytes(&[2.0; 256]);
        let m = Model {
            export: Export::Rotate,
            k: 256,
            eps: 0.0,
            x: &x,
            gate: None,
            s1: vec![-1.0; 256],
            s2: vec![1.0; 256],
            weight: None,
            awq: Some(vec![2.0; 256]),
        };
        assert_eq!(f32s(&m.row(0).rot)[0], -16.0);
    }

    #[test]
    fn sigmoid_range_selects_and_midpoint() {
        // `expf` saturates by its range selects (independent of the exp model): 1 / (1 + 0) and 1 / inf.
        assert_eq!(sigmoid(200.0), 1.0);
        assert_eq!(sigmoid(-200.0), 0.0);
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!((sigmoid(1.0) - 0.731_058_6).abs() < 1e-5);
    }

    #[test]
    fn norm_reduction_orders() {
        // Exact in f32 whatever the order.
        assert_eq!(reduce_sum(&[1.0; 256], Export::Norm), 256.0);
        assert_eq!(reduce_sum(&[1.0; 256], Export::NormOut), 256.0);
        let mut one = [0f32; 256];
        one[100] = 3.0;
        assert_eq!(reduce_sum(&one, Export::Norm), 3.0);
        assert_eq!(reduce_sum(&one, Export::NormOut), 3.0);
        // The two orders differ on wide-range partials (the generator's doc: not interchangeable).
        let mut rng = Xs64::new(9);
        let differing = (0..64)
            .filter(|_| {
                let p: [f32; 256] =
                    std::array::from_fn(|_| rng.unit() * 2f32.powi((rng.next() % 24) as i32));
                reduce_sum(&p, Export::Norm).to_bits() != reduce_sum(&p, Export::NormOut).to_bits()
            })
            .count();
        assert!(differing > 0, "no input separated the two reduction orders");
        // mean 1, eps 0: rms = rsqrt(1).
        assert!((rms_scale(&[1.0; 256], Export::Norm, 0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn sampled_reference_agrees_with_the_full_one_on_its_rows() {
        for export in ["rotate", "norm_out"] {
            let full = c8(export, "rows=64,K=2048");
            let sampled = C8::build(
                "rows=64,K=2048",
                &mut knobs(&[("export", export), ("ref-rows", "sample:4")]),
            )
            .unwrap();
            let (inputs, outs) = (full.inputs(), full.outputs());
            let (f, s) = (
                full.reference(&inputs, &outs).unwrap(),
                sampled.reference(&inputs, &outs).unwrap(),
            );
            assert_eq!(f.outputs.len(), s.outputs.len());
            for (f, s) in f.outputs.iter().zip(&s.outputs) {
                let (f, s) = (f.as_ref().unwrap(), s.as_ref().unwrap());
                assert!(
                    s.covered_bytes() < f.covered_bytes(),
                    "{export}: sample must be partial"
                );
                assert!(s.covered_bytes() > 0);
                for r in &s.ranges {
                    assert_eq!(f.bytes[r.clone()], s.bytes[r.clone()], "{export}");
                }
            }
        }
        // Rows 0..16 and the last 16 plus the spread rows 0, 16, 32, 48 (duplicates dropped, sorted).
        let want: Vec<usize> = (0..=16).chain([32]).chain(48..64).collect();
        assert_eq!(sample_rows(64, 4), want);
        assert_eq!(sample_rows(5, 2), [0, 1, 2, 3, 4]);
    }

    const TWIN: &str = include_str!("../../../peacemaker-isa/src/kernels/iu4_c8_producer.rip");

    fn with_rip(export: &str, text: &str) -> C8 {
        let mut w = c8(export, "");
        w.rip = Some(("twin.rip".into(), text.into()));
        w
    }

    #[test]
    fn rip_twin_is_the_candidate_of_every_export_and_identical_to_the_generator() {
        for export in ["rotate", "norm", "sigmoid_rotate", "norm_out"] {
            let o = with_rip(export, TWIN).candidate().unwrap();
            assert_eq!(o.exports, vec![o.launch_symbol.clone()], "{export}");
            assert_eq!(o.desc["rip"]["identical_to_generator"], true, "{export}");
            assert_eq!(o.desc["rip"]["variant"], export);
            assert_eq!(&o.bytes[..4], b"\x7fELF");
        }
    }

    #[test]
    fn rip_moving_the_prefetch_keeps_the_abi_and_never_claims_identity() {
        let edited = TWIN.replace(
            "    c8::norm_prefetch(b)?;\n    c8::norm_sumsq(b)?;\n",
            "    c8::norm_sumsq(b)?;\n    c8::norm_prefetch(b)?;\n",
        );
        assert_ne!(edited, TWIN);
        let o = with_rip("norm", &edited).candidate().unwrap();
        assert_eq!(o.desc["rip"]["identical_to_generator"], false);
        let e = with_rip("norm", "fn emit(").candidate().unwrap_err();
        assert!(!e.is_empty());
    }

    #[test]
    fn emu_runs_the_small_shape_bit_exact_or_refuses() {
        // The legacy exports are covered by the repo's emulator tests (`tests/iu4_c8_producer.rs`): they
        // must run and match. The folded exports are proven on the card; the emulator may refuse them.
        for (tag, _) in EXPORTS {
            let w = new("rows=8", &mut knobs(&[("export", tag)])).unwrap();
            match w.emu() {
                Value::Object(o) => {
                    assert_eq!(o["ran"], true, "{tag}");
                    assert_eq!(o["bit_exact"], true, "{tag}: {o:?}");
                    assert_eq!(o["mismatching_elements"], 0, "{tag}");
                    assert_eq!(o["shape"]["rows"], 8);
                    assert!(o["reference"].as_str().unwrap().starts_with("cpu_c8_"));
                }
                Value::String(s) => {
                    assert!(s.starts_with("refused ("), "{tag}: {s}");
                    assert!(!s.contains(" | "), "{tag}: {s}");
                    assert!(
                        !matches!(tag, "rotate" | "norm"),
                        "{tag} is covered by the emulator tests but was refused: {s}"
                    );
                }
                other => panic!("{tag}: {other:?}"),
            }
        }
    }
}
