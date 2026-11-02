// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! `--kernel gdn_chunk_prep_pm`: the certified gfx1201 PM twin of the A3B GDN chunk-scan
//! preamble `gdn_chunk_prep` (`peacemaker-isa emit --kernel gdn_chunk_prep_pm`, generator
//! `crates/peacemaker-isa/src/kernels/gdn_chunk_prep.rs`, `GdnChunkPrepSpec::A3B`): causal
//! width-4 conv + SiLU over the projected `[T, 8192]` rows, q/k L2 norm, the C64 gate
//! prefix scan and the f16 q/k/v planes, with the incumbent's ABI byte for byte.
//!
//! The module is the one export `gdn_chunk_prep_pm_gfx1201` (no `--epi` variants, no export
//! choice). Candidate = the native-writer link of the emitted text (the `emit --co` path);
//! baseline = the shipped `rdna_compute::gdn_chunk_prep_pm::IMAGE`
//! (`kernels/gdn_chunk_prep_pm_gfx1201.hxaco`) launched as production does.
//!
//! # Knobs
//! None. The generator has a single geometry (16 key heads, 32 value heads, head dim 128)
//! and the call site fixes the scalars: `q_scale = 1 / sqrt(128)`
//! (`hipfire-dispatch/src/pipeline/batched_moe.rs:851`) and `eps = config.norm_eps` (1e-6,
//! the Qwen3.5 `rms_norm_eps` default `hipfire-arch-qwen35/src/qwen35/config.rs:808-810`; the
//! byte oracle's `EPS`, `examples/gdn_chunk_prep_pm_oracle.rs:56`). Both are recorded in the
//! shape JSON, not tunable.
//!
//! # Shape grammar
//! `[a3b:NAME][,t=ROWS]`; the default is `a3b:t4096`. `t=` overrides the preset's rows; the
//! launcher admits `1..=8192` (`gdn_chunk_prep_pm::admit`). Presets ([`PRESETS`]) cite their
//! call-site source:
//!
//! | preset  | rows | source |
//! |---------|------|--------|
//! | `t1`    | 1    | launcher floor, ring T<3 branch (`gdn_chunk_prep_pm.rs:28`); not reached by the call site |
//! | `t63`   | 63   | ragged tail below the call site's floor (oracle cells, `..._oracle.rs:210`) |
//! | `t64`   | 64   | smallest row count the MoE chunk scan admits (`qwen35/prefill.rs:5684-5688`, `n >= 64`) |
//! | `t65`   | 65   | one row past a C64 chunk, single segment (`prefill.rs:5686`) |
//! | `t129`  | 129  | ragged two-chunk tail (oracle cells) |
//! | `t384`  | 384  | default gfx1201 prefill chunk (`prefill.rs:225`) |
//! | `t512`  | 512  | FP8 gfx1201 prefill chunk, largest single segment (`prefill.rs:198`, `:5686`) |
//! | `t4096` | 4096 | eight 512-row commit segments (`prefill.rs:5687`, `commit_stride == Some(512)`) |
//! | `t4160` | 4160 | 4096 + a 64-row tail (`prefill.rs:5687`, `(64..512).contains(n % 512)`) |
//! | `t8191` | 8191 | launcher max - 1, tail 511 |
//! | `t8192` | 8192 | launcher `MAX_ROWS` (`gdn_chunk_prep_pm.rs:29`) |
//!
//! `fn:` presets are refused: Flash-Next has no GDN chunk-scan prep of this form (it runs
//! 48 value heads, `tests/fixtures/jit-g0/header-flash-next.json:89-92`, and the twin only
//! admits 16/32 heads, `gdn_chunk_prep_pm.rs:92-103`).
//!
//! # Launch (`Gpu::gdn_chunk_prep_pm`, `gdn_chunk_prep_pm.rs:209-271`)
//! Ten pointers `x, conv_w, conv_state, a, b, dt_bias, a_log, q, k, v`, then `T` i32,
//! `q_scale` f32, `eps` f32, `n_key_heads` u32, `n_value_heads` u32 (100 bytes); grid
//! `(ceil(T / 64), 8)`, block 256, no LDS. `a`, `b` and `conv_state` are in-place outputs
//! restored from their input values before every launch; `q`, `k`, `v` are F16 planes.
//!
//! # Reference
//! None (`reference()` reports why): the pass criterion is baseline identity of all six
//! outputs. `--emu` emulates the candidate and the shipped object at 65 rows and compares
//! bytes, or reports the emulator's refusal.

use crate::common::{
    emitted_object, emu_run, f32_unit_bytes, no_leftover, put_f32, put_i32, put_ptr, put_u32,
    read_rip, refused, rip_object, shipped_object, take_dim, Elem, EmuBuf, RipSource, ShapeSpec,
    Xs64,
};
use crate::prod::{self, Expect, OutBuf, Producer, Ptrs};
use crate::workload::{Knobs, Launch, Object, Result, Workload};
use peacemaker_isa::kernels::gdn_chunk_prep::{self, GdnChunkPrepSpec};
use peacemaker_isa::Arch;
use rdna_compute::gdn_chunk_prep_pm::{
    admit, grid, BLOCK, HEAD_DIM, IMAGE, KERNARG_BYTES, KEY_HEADS, SYMBOL, VALUE_HEADS,
};
use serde_json::{json, Value};

/// Channels of one projected row: q, k and v heads (`gdn_chunk_prep_pm.rs:37`).
const CHANNELS: usize = (2 * KEY_HEADS + VALUE_HEADS) * HEAD_DIM;
/// Conv taps and ring depth of the width-four causal conv (`gdn_chunk_prep_pm.rs:39-40`).
const CONV_TAPS: usize = 4;
const CONV_RING: usize = CONV_TAPS - 1;
/// `eps` of the call site (`config.norm_eps`, Qwen3.5 default 1e-6).
const EPS: f32 = 1e-6;
/// Rows of the `--emu` run: two C64 chunks with a one-row tail.
const EMU_ROWS: usize = 65;
/// Shape of an empty `--shape`.
const DEFAULT_PRESET: &str = "t4096";
/// Uniform half-width of unit variance.
const SQRT3: f32 = 1.732_050_8;

const INPUT_NAMES: [&str; 4] = ["x", "conv_w", "dt_bias", "a_log"];
const SEED_X: u64 = 0x6764_6e01;
const SEED_W: u64 = 0x6764_6e02;
const SEED_DT: u64 = 0x6764_6e03;
const SEED_ALOG: u64 = 0x6764_6e04;
const SEED_STATE: u64 = 0x6764_6e05;
const SEED_A: u64 = 0x6764_6e06;
const SEED_B: u64 = 0x6764_6e07;

/// `(name, rows, source)` of the `a3b:` presets.
const PRESETS: [(&str, usize, &str); 11] = [
    (
        "t1",
        1,
        "launcher floor, ring T<3 branch: crates/rdna-compute/src/gdn_chunk_prep_pm.rs:28 (the call site needs n >= 64)",
    ),
    (
        "t63",
        63,
        "ragged tail below the call site's floor: examples/gdn_chunk_prep_pm_oracle.rs:210",
    ),
    (
        "t64",
        64,
        "smallest admitted chunk-scan rows: crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:5684-5688",
    ),
    (
        "t65",
        65,
        "single segment one row past a C64 chunk: prefill.rs:5686, oracle cell :210",
    ),
    (
        "t129",
        129,
        "ragged two-chunk tail: examples/gdn_chunk_prep_pm_oracle.rs:210",
    ),
    (
        "t384",
        384,
        "default gfx1201 prefill chunk: prefill.rs:225 PREFILL_DEFAULT_BATCH_GFX1201",
    ),
    (
        "t512",
        512,
        "FP8 gfx1201 prefill chunk and largest single segment: prefill.rs:198, :5686",
    ),
    (
        "t4096",
        4096,
        "eight 512-row commit segments: prefill.rs:5687 (commit_stride Some(512), n % 512 == 0)",
    ),
    (
        "t4160",
        4160,
        "4096 + a 64-row tail: prefill.rs:5687 ((64..512).contains(n % 512))",
    ),
    (
        "t8191",
        8191,
        "launcher max - 1, tail 511: prefill.rs:5687; oracle cell :210",
    ),
    (
        "t8192",
        8192,
        "launcher MAX_ROWS: crates/rdna-compute/src/gdn_chunk_prep_pm.rs:29; prefill.rs:5687",
    ),
];

const FN_REFUSAL: &str = "gdn_chunk_prep_pm: Flash-Next has no GDN chunk-scan prep of this form: \
it runs 48 value heads (crates/rdna-compute/tests/fixtures/jit-g0/header-flash-next.json:89-92) \
and the twin admits only 16/32 heads (crates/rdna-compute/src/gdn_chunk_prep_pm.rs:92-103); \
use a3b:<name>";

const NO_REFERENCE: &str =
    "no exact CPU reference: the repo's only arithmetic model of this kernel \
(crates/peacemaker-isa/tests/gdn_chunk_prep.rs `model`, lines 594-790) is built on the emulator's \
v_exp/v_log/v_rcp/v_rsq value models (crates/peacemaker-ir/src/emu/trans.rs), which that module \
declares candidate models, not hardware-exact; the incumbent-vs-twin byte proof is the GPU oracle \
examples/gdn_chunk_prep_pm_oracle.rs. A CPU transcription would not be bitwise reproducible, so \
baseline identity of the shipped object decides";

/// `q_scale` of the call site: `1 / sqrt(head_dim)` (`batched_moe.rs:851`).
fn q_scale() -> f32 {
    1.0 / (HEAD_DIM as f32).sqrt()
}

/// `n` f32 values uniform in `[mid - half, mid + half)`, little-endian bytes.
fn f32_affine_bytes(seed: u64, n: usize, mid: f32, half: f32) -> Vec<u8> {
    let mut r = Xs64::new(seed);
    (0..n)
        .flat_map(|_| (mid + r.unit() * half).to_le_bytes())
        .collect()
}

/// The ten kernel pointers in ABI order from the `inputs()` / `outputs()` addresses.
fn abi_pointers(inputs: &[u64], outputs: &[u64]) -> [u64; 10] {
    let (x, w, dt, alog) = (inputs[0], inputs[1], inputs[2], inputs[3]);
    let (state, a, b, q, k, v) = (
        outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5],
    );
    [x, w, state, a, b, dt, alog, q, k, v]
}

/// The launch of `Gpu::gdn_chunk_prep_pm` (`gdn_chunk_prep_pm.rs:120-138`, `:263-270`).
fn launch_of(rows: usize, pointers: [u64; 10]) -> Launch {
    let mut kernarg = Vec::with_capacity(KERNARG_BYTES);
    for p in pointers {
        put_ptr(&mut kernarg, p);
    }
    put_i32(&mut kernarg, rows as i32);
    put_f32(&mut kernarg, q_scale());
    put_f32(&mut kernarg, EPS);
    put_u32(&mut kernarg, KEY_HEADS as u32);
    put_u32(&mut kernarg, VALUE_HEADS as u32);
    Launch {
        kernarg,
        grid: grid(rows),
        block: BLOCK,
        lds: 0,
    }
}

/// Rows and the preset they came from (`(name, source)`; `None` when `t=` overrode it).
fn parse_shape(shape: &str) -> Result<(usize, Option<(&'static str, &'static str)>)> {
    let spec = ShapeSpec::parse(shape)?;
    let name = match &spec.preset {
        None => DEFAULT_PRESET,
        Some((fam, _)) if fam == "fn" => return Err(FN_REFUSAL.into()),
        Some((_, name)) => name.as_str(),
    };
    let (name, rows, source) = *PRESETS.iter().find(|p| p.0 == name).ok_or_else(|| {
        format!(
            "shape: unknown a3b preset `{name}` (one of {})",
            PRESETS.map(|p| p.0).join(", ")
        )
    })?;
    let mut m = spec.resolve(&[("t", rows)]);
    let t = take_dim(&mut m, "t")?;
    no_leftover(&m)?;
    admit(t, KEY_HEADS, VALUE_HEADS)?;
    let overridden = spec.kv.contains_key("t");
    Ok((t, (!overridden).then_some((name, source))))
}

#[cfg(test)]
pub fn new(shape: &str, _knobs: &mut Knobs) -> Result<Box<dyn Workload>> {
    new_with_rip(shape, _knobs, None)
}

pub fn new_with_rip(
    shape: &str,
    _knobs: &mut Knobs,
    rip: Option<&std::path::Path>,
) -> Result<Box<dyn Workload>> {
    let (rows, preset) = parse_shape(shape)?;
    Ok(prod::boxed(Gdn {
        rows,
        preset,
        rip: read_rip(rip)?,
    }))
}

struct Gdn {
    rows: usize,
    preset: Option<(&'static str, &'static str)>,
    /// `--rip`: the candidate is this script's kernel, not the Rust emit.
    rip: RipSource,
}

impl Gdn {
    fn at(rows: usize) -> Self {
        Self {
            rows,
            preset: None,
            rip: None,
        }
    }

    /// Both arms' outputs of one emulated launch, in `outputs()` order.
    fn emulate(&self, object: &Object) -> Result<Vec<Vec<u8>>> {
        let inputs = self.inputs();
        let outs = self.outputs();
        let in_va: Vec<u64> = (0..inputs.len())
            .map(|i| 0x2000_0000 + i as u64 * 0x1000_0000)
            .collect();
        let out_va: Vec<u64> = (0..outs.len())
            .map(|i| 0x6000_0000 + i as u64 * 0x1000_0000)
            .collect();
        let launch = launch_of(self.rows, abi_pointers(&in_va, &out_va));
        let mut bufs: Vec<EmuBuf> = Vec::new();
        for ((name, va), bytes) in INPUT_NAMES.iter().zip(&in_va).zip(inputs) {
            bufs.push(EmuBuf {
                name: *name,
                va: *va,
                bytes,
            });
        }
        for (o, va) in outs.iter().zip(&out_va) {
            bufs.push(EmuBuf {
                name: o.name,
                va: *va,
                bytes: o.init.clone(),
            });
        }
        let mem = emu_run(
            &object.bytes,
            &object.launch_symbol,
            launch.kernarg,
            bufs,
            launch.grid,
            launch.block,
            launch.lds,
        )?;
        outs.iter()
            .zip(&out_va)
            .map(|(o, va)| mem.read(*va, o.init.len()).map(<[u8]>::to_vec))
            .collect()
    }

    fn run_emu(&self) -> Result<Value> {
        let cand = self.emulate(&self.candidate()?)?;
        let shipped = self.emulate(&self.baseline()?)?;
        let outs = self.outputs();
        let mismatching: usize = outs
            .iter()
            .zip(cand.iter().zip(&shipped))
            .map(|(o, (c, s))| {
                let w = o.elem.width();
                c.chunks(w).zip(s.chunks(w)).filter(|(x, y)| x != y).count()
            })
            .sum();
        // A kernel that wrote nothing would match a shipped object that wrote nothing.
        let wrote = outs
            .iter()
            .zip(&cand)
            .filter(|(o, _)| matches!(o.name, "q" | "k" | "v"))
            .all(|(_, c)| c.iter().any(|&b| b != 0));
        Ok(json!({
            "ran": true,
            "bit_exact": mismatching == 0 && wrote,
            "shape": {"t": self.rows, "key_heads": KEY_HEADS, "value_heads": VALUE_HEADS, "head_dim": HEAD_DIM},
            "symbol": SYMBOL,
            "reference": "emulated_shipped_object",
            "mismatching_elements": mismatching,
            "candidate_wrote_qkv": wrote,
        }))
    }
}

impl Producer for Gdn {
    fn kernel(&self) -> &'static str {
        gdn_chunk_prep::KERNEL_ID
    }

    fn shape_json(&self) -> Value {
        let t = self.rows;
        let bytes = |name: &str, n: usize| json!({"name": name, "bytes": n});
        json!({
            "t": t,
            "preset": self.preset.map(|p| format!("a3b:{}", p.0)),
            "preset_source": self.preset.map(|p| p.1),
            "key_heads": KEY_HEADS,
            "value_heads": VALUE_HEADS,
            "head_dim": HEAD_DIM,
            "q_scale": q_scale(),
            "eps": EPS,
            "grid": grid(t),
            "block": BLOCK,
            "kernarg_bytes": KERNARG_BYTES,
            "inputs": [
                bytes("x", t * CHANNELS * 4),
                bytes("conv_w", CHANNELS * CONV_TAPS * 4),
                bytes("dt_bias", VALUE_HEADS * 4),
                bytes("a_log", VALUE_HEADS * 4),
            ],
            "outputs": [
                bytes("conv_state", CHANNELS * CONV_RING * 4),
                bytes("a", t * VALUE_HEADS * 4),
                bytes("b", t * VALUE_HEADS * 4),
                bytes("q", t * KEY_HEADS * HEAD_DIM * 2),
                bytes("k", t * KEY_HEADS * HEAD_DIM * 2),
                bytes("v", t * VALUE_HEADS * HEAD_DIM * 2),
            ],
        })
    }

    fn knobs_json(&self) -> Value {
        json!({
            "export": SYMBOL,
            "spec": "GdnChunkPrepSpec::A3B",
            "hk": KEY_HEADS,
            "hv": VALUE_HEADS,
            "head_dim": HEAD_DIM,
        })
    }

    fn candidate(&self) -> Result<Object> {
        if let Some(rip) = &self.rip {
            return rip_object(
                rip,
                Arch::Gfx1201,
                "chunk_prep",
                gdn_chunk_prep::SYMBOL,
                &gdn_chunk_prep::emit(Arch::Gfx1201, &GdnChunkPrepSpec::A3B)?.s_text,
                self.knobs_json(),
            );
        }
        let (_, text, _) = gdn_chunk_prep::emit_module(Arch::Gfx1201, &GdnChunkPrepSpec::A3B)?;
        emitted_object(
            &text,
            Arch::Gfx1201,
            gdn_chunk_prep::SYMBOL,
            &[gdn_chunk_prep::SYMBOL.to_owned()],
            "peacemaker_isa::kernels::gdn_chunk_prep::emit_module(A3B) + native::assemble \
             (peacemaker-isa emit --kernel gdn_chunk_prep_pm --arch gfx1201 --co)",
            self.knobs_json(),
        )
    }

    fn baseline(&self) -> Result<Object> {
        shipped_object(
            IMAGE,
            "kernels/gdn_chunk_prep_pm_gfx1201.hxaco",
            "crates/rdna-compute/src/gdn_chunk_prep_pm.rs:243 \
             ensure_embedded_kernel(MODULE, IMAGE, SYMBOL), selected by \
             crates/hipfire-dispatch/src/pipeline/batched_moe.rs:499",
            SYMBOL,
            &[SYMBOL],
            json!({"export": SYMBOL}),
        )
    }

    fn inputs(&self) -> Vec<Vec<u8>> {
        vec![
            // The f32 projection `[T, 8192]`: unit-variance activations.
            f32_unit_bytes(SEED_X, self.rows * CHANNELS, SQRT3),
            f32_unit_bytes(SEED_W, CHANNELS * CONV_TAPS, 0.5),
            // The oracle's real-format ranges: dt_bias in [-4, -2), a_log in [0, 2).
            f32_affine_bytes(SEED_DT, VALUE_HEADS, -3.0, 1.0),
            f32_affine_bytes(SEED_ALOG, VALUE_HEADS, 1.0, 1.0),
        ]
    }

    fn outputs(&self) -> Vec<OutBuf> {
        let t = self.rows;
        let state = |name, init| OutBuf {
            name,
            init,
            elem: Elem::F32,
            compare: true,
        };
        vec![
            state(
                "conv_state",
                f32_unit_bytes(SEED_STATE, CHANNELS * CONV_RING, 0.5),
            ),
            state("a", f32_unit_bytes(SEED_A, t * VALUE_HEADS, SQRT3)),
            state("b", f32_unit_bytes(SEED_B, t * VALUE_HEADS, 2.0)),
            OutBuf::zeros("q", t * KEY_HEADS * HEAD_DIM * 2, Elem::F16),
            OutBuf::zeros("k", t * KEY_HEADS * HEAD_DIM * 2, Elem::F16),
            OutBuf::zeros("v", t * VALUE_HEADS * HEAD_DIM * 2, Elem::F16),
        ]
    }

    fn launch(&self, p: &Ptrs) -> Launch {
        launch_of(self.rows, abi_pointers(&p.inputs, &p.outputs))
    }

    fn reference(&self, _inputs: &[Vec<u8>], _outputs: &[OutBuf]) -> Result<Expect> {
        Err(NO_REFERENCE.into())
    }

    fn emu(&self) -> Value {
        // The emulator's refusal names the first opcode it has no semantics for.
        Self::at(EMU_ROWS)
            .run_emu()
            .unwrap_or_else(|e| refused(e.split(" | ").next().unwrap_or(&e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::sha256_hex;
    use rdna_compute::gdn_chunk_prep_pm::MAX_ROWS;

    fn f32s(b: &[u8]) -> Vec<f32> {
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn shape_grammar_and_presets() {
        assert_eq!(parse_shape("").unwrap().0, 4096);
        assert_eq!(parse_shape("a3b:t4096").unwrap().0, 4096);
        assert_eq!(parse_shape("a3b:t65").unwrap().0, 65);
        let (t, preset) = parse_shape("a3b:t4096,t=100").unwrap();
        assert_eq!((t, preset), (100, None), "t= overrides the preset");
        assert_eq!(parse_shape("t=8192").unwrap().0, 8192);
        for (name, rows, source) in PRESETS {
            assert_eq!(name, format!("t{rows}"));
            assert!(!source.is_empty());
            assert!(admit(rows, KEY_HEADS, VALUE_HEADS).is_ok(), "{name}");
            assert_eq!(parse_shape(&format!("a3b:{name}")).unwrap().0, rows);
        }
        assert!(PRESETS.iter().any(|p| p.1 == 8192 && p.1 == MAX_ROWS));
        assert!(PRESETS.iter().all(|p| p.0 != "t0" && p.1 <= MAX_ROWS));
        let fnerr = parse_shape("fn:t4096").unwrap_err();
        assert!(
            fnerr.contains("Flash-Next") && fnerr.contains("48 value heads"),
            "{fnerr}"
        );
        for bad in [
            "a3b:t5",
            "t=0",
            "t=8193",
            "t=x",
            "a3b:t64,rows=1",
            "a3b:t64,a3b:t65",
        ] {
            assert!(parse_shape(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn kernarg_layout_and_grid() {
        let ptrs: [u64; 10] = std::array::from_fn(|i| 0x1000 * (i as u64 + 1));
        let l = launch_of(8191, ptrs);
        assert_eq!(l.kernarg.len(), 100);
        for (i, p) in ptrs.iter().enumerate() {
            assert_eq!(l.kernarg[i * 8..i * 8 + 8], p.to_le_bytes(), "pointer {i}");
        }
        assert_eq!(l.kernarg[80..84], 8191i32.to_le_bytes());
        assert_eq!(l.kernarg[84..88], (1.0f32 / 128f32.sqrt()).to_le_bytes());
        assert_eq!(l.kernarg[88..92], 1e-6f32.to_le_bytes());
        assert_eq!(l.kernarg[92..96], 16u32.to_le_bytes());
        assert_eq!(l.kernarg[96..100], 32u32.to_le_bytes());
        assert_eq!((l.grid, l.block, l.lds), ([128, 8, 1], [256, 1, 1], 0));
        for (rows, gx) in [(1, 1), (63, 1), (64, 1), (65, 2), (4096, 64), (8192, 128)] {
            assert_eq!(launch_of(rows, ptrs).grid, [gx, 8, 1], "rows {rows}");
        }
    }

    #[test]
    fn pointer_order_is_the_abi_order() {
        let g = Gdn::at(3);
        let l = g.launch(&Ptrs {
            inputs: vec![1, 2, 3, 4],
            outputs: vec![10, 11, 12, 13, 14, 15],
        });
        let at = |i: usize| u64::from_le_bytes(l.kernarg[i * 8..i * 8 + 8].try_into().unwrap());
        // x, conv_w, conv_state, a, b, dt_bias, a_log, q, k, v
        assert_eq!(
            (0..10).map(at).collect::<Vec<_>>(),
            vec![1, 2, 10, 11, 12, 3, 4, 13, 14, 15]
        );
        assert_eq!(l.kernarg[80..84], 3i32.to_le_bytes());
    }

    #[test]
    fn candidate_is_a_fresh_emit_and_assemble() {
        let g = Gdn::at(64);
        let o = g.candidate().unwrap();
        let (_, text, _) =
            gdn_chunk_prep::emit_module(Arch::Gfx1201, &GdnChunkPrepSpec::A3B).unwrap();
        assert_eq!(
            o.bytes,
            peacemaker_isa::native::assemble(&text, Arch::Gfx1201).unwrap()
        );
        assert_eq!(o.launch_symbol, "gdn_chunk_prep_pm_gfx1201");
        assert_eq!(o.exports, vec!["gdn_chunk_prep_pm_gfx1201"]);
        assert_eq!(o.source["sha256"], sha256_hex(&o.bytes));
        assert_eq!(g.kernel(), "gdn_chunk_prep_pm");
        assert_eq!(g.knobs_json()["hv"], 32);
    }

    #[test]
    fn baseline_is_the_shipped_image() {
        let b = Gdn::at(64).baseline().unwrap();
        assert_eq!(b.bytes, IMAGE);
        assert_eq!(b.launch_symbol, "gdn_chunk_prep_pm_gfx1201");
        assert_eq!(b.source["sha256"], sha256_hex(IMAGE));
        assert!(b.source["source"]
            .as_str()
            .unwrap()
            .contains("gdn_chunk_prep_pm.rs:243"));
        assert!(crate::common::contains(&b.bytes, SYMBOL.as_bytes()));
    }

    #[test]
    fn inputs_and_outputs_are_deterministic_and_in_range() {
        let g = Gdn::at(3);
        let ins = g.inputs();
        assert_eq!(ins, g.inputs());
        assert_eq!(
            ins.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![3 * 8192 * 4, 8192 * 4 * 4, 128, 128]
        );
        for v in ins.iter().flat_map(|b| f32s(b)) {
            assert!(v.is_finite());
        }
        assert!(f32s(&ins[0]).iter().all(|v| v.abs() <= 1.732_051));
        assert!(f32s(&ins[2]).iter().all(|v| (-4.0..=-2.0).contains(v)));
        assert!(f32s(&ins[3]).iter().all(|v| (0.0..=2.0).contains(v)));
        let outs = g.outputs();
        let names: Vec<_> = outs.iter().map(|o| o.name).collect();
        assert_eq!(names, ["conv_state", "a", "b", "q", "k", "v"]);
        let sizes: Vec<_> = outs.iter().map(|o| o.init.len()).collect();
        assert_eq!(
            sizes,
            [
                8192 * 3 * 4,
                3 * 32 * 4,
                3 * 32 * 4,
                3 * 2048 * 2,
                3 * 2048 * 2,
                3 * 4096 * 2
            ]
        );
        assert!(outs.iter().all(|o| o.compare));
        assert_eq!(outs[3].elem, Elem::F16);
        assert_eq!(outs[0].elem, Elem::F32);
        assert!(
            outs[..3].iter().all(|o| o.init.iter().any(|&b| b != 0)),
            "in-place state is non-trivial"
        );
        assert!(outs[3..].iter().all(|o| o.init.iter().all(|&b| b == 0)));
        assert_eq!(outs[0].init, g.outputs()[0].init);
    }

    #[test]
    fn shape_json_matches_the_buffer_sizes() {
        let g = Gdn::at(65);
        let s = g.shape_json();
        assert_eq!(s["grid"], json!([2, 8, 1]));
        let sizes = |key: &str| -> Vec<u64> {
            s[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|b| b["bytes"].as_u64().unwrap())
                .collect()
        };
        assert_eq!(
            sizes("inputs"),
            g.inputs()
                .iter()
                .map(|b| b.len() as u64)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            sizes("outputs"),
            g.outputs()
                .iter()
                .map(|o| o.init.len() as u64)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn reference_reports_why_there_is_none() {
        let g = Gdn::at(64);
        let e = g.reference(&g.inputs(), &g.outputs()).unwrap_err();
        assert!(
            e.contains("v_exp") && e.contains("not hardware-exact"),
            "{e}"
        );
    }

    /// The emulator either runs candidate and shipped object to the same bytes or refuses
    /// by name; it never reports a mismatch.
    const TWIN: &str = include_str!("../../../peacemaker-isa/src/kernels/gdn_chunk_prep.rip");

    #[test]
    fn rip_twin_is_the_candidate_and_identical_to_the_generator() {
        let mut g = Gdn::at(64);
        g.rip = Some(("twin.rip".into(), TWIN.into()));
        let o = g.candidate().unwrap();
        assert_eq!(o.launch_symbol, gdn_chunk_prep::SYMBOL);
        assert_eq!(o.desc["rip"]["identical_to_generator"], true);
        assert_eq!(o.desc["rip"]["variant"], "chunk_prep");
        // A twin edited so the gate loop's block is dropped changes the program but keeps the ABI.
        let edited = TWIN.replace(
            "    prod::wait_all(&mut b)?;\n    prod::label(&mut b, gdn::label(\"end\")?)?;",
            "    prod::label(&mut b, gdn::label(\"end\")?)?;",
        );
        assert_ne!(edited, TWIN);
        g.rip = Some(("edited.rip".into(), edited));
        if let Ok(o) = g.candidate() {
            assert_eq!(o.desc["rip"]["identical_to_generator"], false);
        }
    }

    #[test]
    fn emu_matches_the_shipped_object_or_refuses() {
        let mut k = Knobs::default();
        let w = new("a3b:t65", &mut k).unwrap();
        k.finish().unwrap();
        match w.emu() {
            Value::String(s) => assert!(s.starts_with("refused (") && !s.contains(" | "), "{s}"),
            v => {
                assert_eq!(v["ran"], true, "{v}");
                assert_eq!(v["bit_exact"], true, "{v}");
                assert_eq!(v["reference"], "emulated_shipped_object");
                assert_eq!(v["mismatching_elements"], 0);
            }
        }
    }
}
